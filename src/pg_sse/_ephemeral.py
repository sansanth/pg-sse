"""
Per-key signals that lapse: state that is true only while a client keeps renewing it (someone is typing, someone
has a room open). When it changes or lapses, the key's streams are woken, so a closed tab never leaves a stale
"typing" on another screen.

State is kept in this process's memory, like `hub.wake`: with several workers, a signal set on one is invisible to
the streams on the others, unless it is `shared`. Then each change is also sent as a NOTIFY to the other hubs on the
database, which apply it as if it had been made there. A received signal lapses by its ttl on the receiver's own clock,
so no two machines' clocks are ever compared, and a lost message costs at most one ttl of a stale signal.

Every hub, the sender included, applies the changes in the order Postgres delivers them, so two hubs that change one
signal at the same moment still agree on the outcome. While a hub's own change to a signal is on its way back (a clear
of everyone counts for each who), it skips whatever arrives for that signal first, its own older echoes and other
hubs' changes alike: those come earlier in Postgres's order and would only show for a round trip. After `reset`, the
echoes of the starts and renewals a hub sent before it are ignored.
"""

import heapq
import json
import os
import secrets
import threading
import time
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from ._hub import _EPHEMERAL_PAYLOAD_LIMIT, _check_key, _seconds, check_signal

if TYPE_CHECKING:
    from ._hub import Hub


class Ephemeral:
    """
    Signals per key that lapse unless renewed, announced to the key's streams with `hub.wake`.

    - `name`: what this is for ("typing", "viewing"). It is the default `signal`.
    - `ttl`: seconds a signal lasts without renewal.
    - `signal`: the label `hub.wake` carries, so `wake.signals` contains it. Two instances may share one, to wake
      one stream for either.
    - `grace`: the lapse check runs this long after expiry, so it never finds a signal a hair from expiring.
    - `all_keys`: passed to `hub.wake`. False by default: ephemeral state is per key, and a view of every key has
      nothing to show for it.
    - `clock`: seconds, monotonic. Replace it in tests.
    - `shared`: also send each change to the other hubs on the database through `Hub(publish=...)`, and apply theirs.
      `who` must then be a `str` or an `int` (`TypeError`), and a change too large for a NOTIFY raises `ValueError`,
      both before anything changes.

    Lapses are noticed by `hub.call_later`, which does nothing while the hub is stopped: entries then linger until the
    next `set` or `clear` for their key, and `live` never lists an expired one.
    """

    def __init__(
        self,
        hub: "Hub",
        name: str,
        *,
        ttl: float,
        signal: str | None = None,
        grace: float = 0.25,
        all_keys: bool = False,
        clock: Callable[[], float] = time.monotonic,
        shared: bool = False,
    ) -> None:
        self.hub = hub
        self.name = name
        self.ttl = _seconds("ttl", ttl)
        self.signal = name if signal is None else signal
        check_signal(self.signal)  # at construction, not at the first `set`
        self.grace = _seconds("grace", grace, zero=True)
        self.all_keys = all_keys
        self._clock = clock
        self._lock = threading.Lock()
        # key -> who -> when it lapses, in insertion order. A key with nobody has no entry.
        self._entries: dict[str, dict[str, float]] = {}
        # Bumped by `reset`: a lapse check scheduled before it belongs to a state that no longer exists.
        self._generation = 0
        # Shared only: (key, who) -> when the other hubs' copy lapses, as far as this hub knows. Renewals here move
        # `_entries` on but are not all sent, so the two part ways.
        self._told: dict[tuple[str, Any], float] = {}
        # Shared only: this instance's changes carry its origin and a running number. key -> who (None: everyone)
        # -> the number of the last change sent for them, until it comes back, fails to send, or a ttl has passed.
        # Postgres delivers in commit order, so whatever arrives for them before it is older and would only show for
        # a round trip: it is skipped. `_expiry` is a heap of (when, number, key, who).
        self._pid = os.getpid()
        self._origin = secrets.token_hex(8)
        self._number = 0
        self._reset_number = 0  # the last change sent before `reset`: an echo up to it that starts a signal is ignored
        self._sent: dict[str, dict[Any, int]] = {}
        self._expiry: list[tuple[float, int, str, Any]] = []
        self.shared = shared
        if shared:
            hub._share(self)

    def set(self, key: str, who: str, on: bool = True) -> bool:
        """
        Renews or starts (`on`) or removes (`not on`) `who`'s signal on `key`. Safe from any thread.

        Returns whether what a reader sees changed, and wakes the key's streams only then. A renewal changes nothing.
        Removing wakes if there was an entry, even one that lapsed a moment ago: someone may have read it before it
        lapsed.
        """
        _check_key(key, allow_none=False)
        if self.shared:
            self._check_shared(key, who)
        if not on:
            return self._remove(key, who, send=self.shared)
        return self._start(key, who, self.ttl, send=self.shared)

    def clear(self, key: str, who: str | None = None) -> bool:
        """Removes `who`'s signal on `key`, or everyone's. Returns and wakes as `set(..., on=False)` does."""
        _check_key(key, allow_none=False)
        if who is None:
            if self.shared:
                self._check_shared(key, None, everyone=True)
            return self._remove_all(key, send=self.shared)
        if self.shared:
            self._check_shared(key, who)
        return self._remove(key, who, send=self.shared)

    def reset(self) -> None:
        """
        Forgets every key and signal, received ones too, without waking any stream or telling other hubs, and makes the
        lapse checks already scheduled do nothing; the echoes of the starts and renewals this instance sent before it
        are ignored, while other hubs' changes still apply. For tests: a fixture's clean slate.
        """
        with self._lock:
            self._reset_number = self._number
            self._entries.clear()
            self._told.clear()
            self._sent.clear()
            self._expiry.clear()
            self._generation += 1

    def live(self, key: str) -> list[str]:
        """Whoever has not lapsed on `key` right now, in the order they started, even if a lapse check has not run."""
        now = self._clock()
        with self._lock:
            return [who for who, until in self._entries.get(key, {}).items() if until > now]

    def _start(self, key: str, who: Any, ttl: float, *, send: bool, echo: int | None = None) -> bool:
        now = self._clock()
        with self._lock:
            # An own start sent before a `reset` would bring back what it forgot. A removal still applies: in Postgres's
            # order it may come after another hub's start, as it does on the other hubs.
            if echo is not None and (0 < echo <= self._reset_number or self._superseded(key, who, echo)):
                return False
            users = self._entries.setdefault(key, {})
            previous = users.get(who)
            changed = previous is None or previous <= now
            users[who] = now + ttl
            generation = self._generation
            # A renewal is sent only once the other hubs' copy has less than half its ttl left, so a signal renewed on
            # every keystroke costs one NOTIFY per half ttl and still never lapses there. Sent under the lock, so
            # changes to one key leave in the order they were made.
            if send:
                if changed or self._told.get((key, who), now) - now < ttl / 2:
                    self._send(key, who, True)
                    self._told[key, who] = now + ttl
            elif self.shared:  # received: every hub got it at about the same time
                self._told[key, who] = now + ttl
        if changed:
            self._wake(key)
        # One timer per renewal, not per key: `call_later` does nothing while the hub is stopped, so a "pending"
        # flag set then would never be cleared. An earlier timer finds the signal renewed and leaves it alone.
        self.hub.call_later(ttl + self.grace, self._check, key, generation)
        return changed

    def _remove(self, key: str, who: Any, *, send: bool, echo: int | None = None) -> bool:
        with self._lock:
            if echo is not None and self._superseded(key, who, echo):
                return False
            if send:  # even with nothing here to remove: another hub may have missed the set
                self._send(key, who, False)
            self._told.pop((key, who), None)
            users = self._entries.get(key)
            if users is None or who not in users:
                return False
            del users[who]
            if not users:
                del self._entries[key]
        self._wake(key)
        return True

    def _remove_all(self, key: str, *, send: bool, echo: int | None = None) -> bool:
        with self._lock:
            if send:
                self._send(key, None, False)
            kept = {} if echo is None else self._newer(key, echo)
            users = self._entries.get(key, {})
            removed = [who for who in users if who not in kept]
            for who in removed:
                del users[who]
                self._told.pop((key, who), None)
            if not users:
                self._entries.pop(key, None)
        if removed:
            self._wake(key)
        return bool(removed)

    def _check_shared(self, key: str, who: object, *, everyone: bool = False) -> None:
        """Refuses a change that could not be sent, before anything changes."""
        if not everyone and not _is_who(who):
            raise TypeError(f"pg_sse: a shared Ephemeral's who must be a str or an int, got {type(who).__name__}")
        # A character takes at most 6 bytes in the payload, so only a long key or who has to be built to be measured;
        # most `set`s are renewals that send nothing.
        if 6 * (len(self.name) + len(key) + len(str(who))) + 150 <= _EPHEMERAL_PAYLOAD_LIMIT:
            return
        size = len(self._payload(key, who, False, self._number + 1).encode())
        if size > _EPHEMERAL_PAYLOAD_LIMIT:
            raise ValueError(
                f"pg_sse: this change of {self.name!r} is {size} bytes as a NOTIFY payload, over the "
                f"{_EPHEMERAL_PAYLOAD_LIMIT} a shared Ephemeral sends; use a shorter key or who"
            )

    def _send(self, key: str, who: Any, on: bool) -> None:
        """Under the lock, so changes leave in the order they were made: sends one, and waits for it to come back."""
        payload = self._message(key, who, on)
        number = self._number
        if not self.hub._send_ephemeral(payload, lambda: self._unsent(key, who, number)):
            self._forget(key, who, number)

    def _message(self, key: str, who: Any, on: bool) -> str:
        """Under the lock: the payload of a change about to be sent."""
        self._here()
        self._expire()
        self._number += 1
        self._sent.setdefault(key, {})[who] = self._number
        heapq.heappush(self._expiry, (self._clock() + self.ttl, self._number, key, who))
        return self._payload(key, who, on, self._number)

    def _payload(self, key: str, who: Any, on: bool, number: int) -> str:
        return json.dumps(
            {"v": 1, "e": self.name, "k": key, "w": who, "on": on, "ttl": self.ttl, "o": self._origin, "n": number},
            ensure_ascii=False,
            separators=(",", ":"),
        )

    def _superseded(self, key: str, who: Any, echo: int) -> bool:
        """
        Under the lock, for a received change of `who` on `key`: whether this instance has sent a newer one for them
        that has not come back yet. `echo` is the received change's number if it is this instance's own, else 0.
        """
        self._expire()
        if not (sent := self._sent.get(key)):
            return False
        pending = max(sent.get(who, 0), sent.get(None, 0))
        if echo:
            self._forget(key, who, echo)  # if it was the last change sent for them, it came back
        return pending > echo

    def _newer(self, key: str, echo: int) -> dict[Any, int]:
        """
        Under the lock, for a received clear of everyone on `key`: who this instance has sent a newer change for, which
        the clear must leave alone. `echo` as in `_superseded`; an echo also forgets the changes it covers.
        """
        self._expire()
        newer = {who: number for who, number in self._sent.get(key, {}).items() if number > echo}
        if echo:
            if newer:
                self._sent[key] = newer
            else:
                self._sent.pop(key, None)
        return newer

    def _expire(self) -> None:
        """Under the lock: forgets the changes sent over a ttl ago. Their echo is late or lost; nothing waits for it."""
        while self._expiry and self._expiry[0][0] <= self._clock():
            _, number, key, who = heapq.heappop(self._expiry)
            self._forget(key, who, number)

    def _forget(self, key: str, who: Any, number: int) -> None:
        """Under the lock: stops waiting for change `number` of `who` on `key`, unless a later one was sent since."""
        sent = self._sent.get(key)
        if sent is None or sent.get(who) != number:
            return
        del sent[who]
        if not sent:
            del self._sent[key]

    def _unsent(self, key: str, who: Any, number: int) -> None:
        """From the publisher, when `publish` raised: that change will not come back."""
        with self._lock:
            self._forget(key, who, number)

    def _listening(self) -> None:
        """
        From the listener, once it LISTENs (again): changes sent before that will not come back, and the ones sent
        since cannot be told apart from them.
        """
        with self._lock:
            self._sent.clear()
            self._expiry.clear()

    def _here(self) -> None:
        """
        Under the lock: a process forked from the one that made this instance gets an origin of its own, or it would
        take its parent's and its siblings' changes for its own.
        """
        if (pid := os.getpid()) != self._pid:
            self._pid = pid
            self._origin = secrets.token_hex(8)
            self._sent.clear()
            self._expiry.clear()

    def _receive(self, message: dict[str, Any]) -> bool:
        """Applies a hub's change as a local one, sending it nowhere. False, changing nothing, if malformed."""
        key, who, on, ttl = message.get("k"), message.get("w"), message.get("on"), message.get("ttl")
        if not isinstance(key, str) or not isinstance(on, bool):
            return False
        number = message.get("n")
        with self._lock:
            self._here()
            own = message.get("o") == self._origin
        echo = number if own and type(number) is int else 0
        if who is None and not on:
            self._remove_all(key, send=False, echo=echo)
            return True
        if not _is_who(who):
            return False
        if not on:
            self._remove(key, who, send=False, echo=echo)
            return True
        try:
            seconds = _seconds("ttl", ttl)
        except ValueError:
            return False
        self._start(key, who, seconds, send=False, echo=echo)
        return True

    def _check(self, key: str, generation: int) -> None:
        """Runs on the hub's loop `ttl + grace` after a renewal: drops what lapsed, and says so only if it dropped."""
        now = self._clock()
        with self._lock:
            if generation != self._generation:
                return  # scheduled before a `reset`
            users = self._entries.get(key)
            if not users:
                return
            lapsed = [who for who, until in users.items() if until <= now]
            for who in lapsed:
                del users[who]
                self._told.pop((key, who), None)
            if not users:
                del self._entries[key]
        if lapsed:
            self._wake(key)

    def _wake(self, key: str) -> None:
        self.hub.wake(key, self.signal, all_keys=self.all_keys)  # outside the lock: it only posts to the loop


def _is_who(who: object) -> bool:
    """A `who` a shared `Ephemeral` can send: JSON gives it back as it was, and equal to itself."""
    return isinstance(who, str | int) and not isinstance(who, bool)
