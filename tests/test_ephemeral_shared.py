import asyncio
import json
import logging
import os
import threading
from contextlib import AsyncExitStack
from unittest import mock

import pytest
from helpers import execute, make_hub, until

from pg_sse import _hub as hub_module

pytestmark = pytest.mark.anyio


class Notify:
    """A `publish` that sends the NOTIFY on a connection of its own and keeps what it sent."""

    def __init__(self, dsn: str) -> None:
        self.dsn = dsn
        self.sent: list[dict] = []

    def __call__(self, channel: str, payload: str) -> None:
        # Kept first: once sent, the other hub may act on it, and the test go on, before this thread runs again.
        self.sent.append(json.loads(payload))
        execute(self.dsn, "SELECT pg_notify(%s, %s)", (channel, payload))


class Clock:
    def __init__(self) -> None:
        self.now = 100.0

    def __call__(self) -> float:
        return self.now


def two_hubs(db, **kwargs):
    """Two hubs on one database, as two processes would have, each with a shared `typing` (ttl 8)."""
    notify = Notify(db.dsn)
    a, b = make_hub(db, publish=notify), make_hub(db, publish=notify)
    typing_a = a.ephemeral("typing", **{"ttl": 8, "shared": True, **kwargs})
    typing_b = b.ephemeral("typing", ttl=8, shared=True)
    return a, b, typing_a, typing_b, notify


async def running(stack: AsyncExitStack, *hubs) -> None:
    for hub in hubs:
        await stack.enter_async_context(hub.lifespan())


async def woken(subscription, timeout: float = 0.2) -> bool:
    return await subscription.wait(timeout) is not None


async def test_a_signal_set_on_one_hub_wakes_the_keys_streams_on_the_other(db):
    a, b, typing_a, typing_b, notify = two_hubs(db)
    viewing_b = b.ephemeral("viewing", ttl=8, signal="presence", shared=True)
    viewing_a = a.ephemeral("viewing", ttl=8, shared=True)
    async with AsyncExitStack() as stack:
        await running(stack, a, b)
        subscription = b.subscribe("room")
        typing_a.set("room", "ann")
        assert await subscription.wait(5) == [("wake", "room", "typing")]
        assert typing_b.live("room") == ["ann"]
        viewing_a.set("room", "ann")
        assert await subscription.wait(5) == [("wake", "room", "presence")]  # the receiving instance's signal
        typing_a.clear("room", "ann")
        assert await subscription.wait(5) == [("wake", "room", "typing")]
        assert typing_b.live("room") == []
        assert viewing_b.live("room") == ["ann"]
        assert [m["on"] for m in notify.sent] == [True, True, False]


async def test_clearing_a_key_on_one_hub_clears_it_on_the_other_and_set_off_does_too(db):
    a, b, typing_a, typing_b, _ = two_hubs(db)
    async with AsyncExitStack() as stack:
        await running(stack, a, b)
        subscription = b.subscribe("room")
        typing_a.set("room", "ann")
        typing_a.set("room", "bob")
        await until(lambda: typing_b.live("room") == ["ann", "bob"])
        subscription.take()
        typing_a.set("room", "ann", on=False)
        await until(lambda: typing_b.live("room") == ["bob"])
        typing_a.clear("room")
        await until(lambda: typing_b.live("room") == [])
        assert await subscription.wait(1) == [("wake", "room", "typing")]  # both removals, one wake each


async def test_a_clear_reaches_a_signal_the_other_hub_set_itself(db):
    a, b, typing_a, typing_b, _ = two_hubs(db)
    async with AsyncExitStack() as stack:
        await running(stack, a, b)
        typing_b.set("room", "ann")  # ann's keystroke went to B, her message to A
        await until(lambda: typing_a.live("room") == ["ann"])
        typing_a.clear("room", "ann")
        await until(lambda: typing_b.live("room") == [])


async def test_a_received_signal_lapses_after_the_senders_ttl_on_the_receivers_clock(db):
    a, b, typing_a, typing_b, _ = two_hubs(db)
    typing_a.ttl = 0.3
    typing_b.ttl = 60  # the sender's ttl counts, not the receiver's
    async with AsyncExitStack() as stack:
        await running(stack, a, b)
        subscription = b.subscribe("room")
        typing_a.set("room", "ann")
        assert await subscription.wait(5) == [("wake", "room", "typing")]
        assert typing_b.live("room") == ["ann"]
        assert await subscription.wait(5) == [("wake", "room", "typing")]  # the lapse
        assert typing_b.live("room") == []


async def test_a_renewal_is_sent_only_once_less_than_half_its_ttl_is_left(db):
    clock = Clock()
    a, b, typing_a, typing_b, notify = two_hubs(db, clock=clock)
    async with AsyncExitStack() as stack:
        await running(stack, a, b)
        typing_a.set("room", "ann")  # sent: new
        clock.now += 3
        typing_a.set("room", "ann")  # 5 s left of 8 on B: not sent
        clock.now += 2
        typing_a.set("room", "ann")  # renewed here 2 s ago, but B's copy has 3 s left: sent
        clock.now += 1
        typing_a.set("room", "ann")  # B was told a second ago: not sent
        typing_a.set("room", "bob")  # a new who: always sent
        await until(lambda: len(notify.sent) == 3)
        assert [(m["w"], m["on"]) for m in notify.sent] == [("ann", True), ("ann", True), ("bob", True)]
        await until(lambda: typing_b.live("room") == ["ann", "bob"])


async def test_two_hubs_that_change_one_signal_at_once_agree_on_the_outcome(db):
    notify = Notify(db.dsn)
    held = threading.Event()

    def held_back(channel: str, payload: str) -> None:
        held.wait(5)
        notify(channel, payload)

    a, b = make_hub(db, publish=notify), make_hub(db, publish=held_back)
    typing_a = a.ephemeral("typing", ttl=8, shared=True)
    typing_b = b.ephemeral("typing", ttl=8, shared=True)
    async with AsyncExitStack() as stack:
        await running(stack, a, b)
        with mock.patch.object(typing_b, "_receive", wraps=typing_b._receive) as received:
            typing_b.clear("room", "ann")  # applied on B at once, sent after A's set
            typing_a.set("room", "ann")
            await until(lambda: received.call_count == 1)
        assert typing_b.live("room") == []  # A's set came first: B's clear, on its way, overrides it
        held.set()
        await until(lambda: typing_a.live("room") == [] and typing_b.live("room") == [])  # B's clear came last
        await asyncio.sleep(0.1)
        assert typing_a.live("room") == typing_b.live("room") == []
        assert a.stats().ephemeral_rejected_total == b.stats().ephemeral_rejected_total == 0


async def test_a_hubs_own_change_comes_back_and_leaves_it_where_it_was(db):
    a, b, typing_a, typing_b, _ = two_hubs(db)
    async with AsyncExitStack() as stack:
        await running(stack, a, b)
        subscription = a.subscribe("room")
        typing_a.set("room", "ann")
        await until(lambda: typing_b.live("room") == ["ann"])
        await asyncio.sleep(0.1)  # A's echo arrived too: a renewal, no second wake
        assert await subscription.wait(1) == [("wake", "room", "typing")]
        assert not await woken(subscription)
        assert typing_a.live("room") == ["ann"]


def captured(db, **kwargs):
    """A hub whose `publish` only keeps the payloads, to be handed back by the test, and its shared `typing`."""
    sent: list[str] = []
    hub = make_hub(db, publish=lambda channel, payload: sent.append(payload))
    return hub, hub.ephemeral("typing", ttl=8, shared=True, **kwargs), sent


@pytest.mark.parametrize(
    ("changes", "live"),
    [
        ("on off", []),
        ("off on", ["ann"]),
        ("on clear", []),
        ("on clear on", ["ann"]),
        ("clear on off", []),
    ],
)
async def test_the_echo_of_a_change_that_was_followed_by_a_newer_one_is_skipped(db, changes, live):
    hub, typing, sent = captured(db)
    for change in changes.split():
        if change == "clear":
            typing.clear("room")
        else:
            typing.set("room", "ann", change == "on")
    await until(lambda: len(sent) == len(changes.split()))
    for payload in sent:
        hub._receive_ephemeral(payload)
        assert typing.live("room") == live
    assert hub.stats().ephemeral_rejected_total == 0
    hub.stop()


async def test_the_echo_of_a_clear_removes_the_others_and_keeps_whom_this_hub_set_again(db):
    a, typing_a, sent_a = captured(db)
    b, typing_b, sent_b = captured(db)
    typing_a.set("room", "ann")
    typing_b.set("room", "bob")
    typing_a.clear("room")
    typing_a.set("room", "ann")
    await until(lambda: len(sent_a) == 3 and len(sent_b) == 1)
    # bob's set committed before A's clear, so A never shows him; A shows ann throughout, as it did locally
    steps = [(sent_a[0], ["ann"], ["bob", "ann"]), (sent_b[0], ["ann"], ["bob", "ann"])]
    steps += [(sent_a[1], ["ann"], []), (sent_a[2], ["ann"], ["ann"])]
    for payload, live_a, live_b in steps:
        a._receive_ephemeral(payload)
        b._receive_ephemeral(payload)
        assert (typing_a.live("room"), typing_b.live("room")) == (live_a, live_b)
    a.stop()
    b.stop()


async def test_two_hubs_that_change_one_signal_end_in_postgres_order(db):
    a, typing_a, sent_a = captured(db)
    b, typing_b, sent_b = captured(db)
    typing_a.set("room", "ann", True)
    typing_b.set("room", "ann", False)
    await until(lambda: len(sent_a) == len(sent_b) == 1)
    # B's NOTIFY committed first: A, whose set is on its way, never shows B's older clear
    for payload, live_b in ((sent_b[0], []), (sent_a[0], ["ann"])):
        a._receive_ephemeral(payload)
        b._receive_ephemeral(payload)
        assert (typing_a.live("room"), typing_b.live("room")) == (["ann"], live_b)
    a.stop()
    b.stop()


async def test_a_change_whose_echo_never_comes_is_forgotten_after_a_ttl(db):
    clock = Clock()
    hub, typing, sent = captured(db, clock=clock)
    for who in ("ann", "bob", "cy"):
        typing.set("room", who)
    typing.clear("lobby")
    clock.now += 8
    typing.set("room", "dee")
    assert typing._sent == {"room": {"dee": 5}}
    await until(lambda: len(sent) == 5)
    hub._receive_ephemeral(sent[0])  # too late to be told apart from anyone else's: applied
    assert sorted(typing.live("room")) == ["ann", "dee"]
    hub.stop()


async def test_a_change_made_while_the_publisher_drains_is_sent_after_what_it_drains(db):
    released = threading.Event()
    sent: list[str] = []

    def publish(channel: str, payload: str) -> None:
        who = json.loads(payload)["w"]
        if who == "ann":
            released.wait(5)
        sent.append(who)

    hub = make_hub(db, publish=publish)
    typing = hub.ephemeral("typing", ttl=8, shared=True)
    typing.set("room", "ann")
    stopping = threading.Thread(target=hub._stop_publisher)
    stopping.start()
    await until(lambda: hub._publisher is None)
    typing.set("room", "bob")  # a second publisher, while the first still sends ann
    await asyncio.sleep(0.1)
    released.set()
    stopping.join()  # and the stop waits for that one too
    assert sent == ["ann", "bob"]
    hub.stop()


async def test_a_change_that_was_not_sent_holds_nothing_back(db):
    def publish(channel: str, payload: str) -> None:
        raise ConnectionError("database gone")

    hub = make_hub(db, publish=publish)
    typing = hub.ephemeral("typing", ttl=8, shared=True)
    typing.set("room", "ann", on=False)
    await until(lambda: typing._sent == {})
    other = {"v": 1, "e": "typing", "k": "room", "w": "ann", "on": True, "ttl": 8}
    hub._receive_ephemeral(json.dumps(other))  # no change of this hub's is on its way to override it
    assert typing.live("room") == ["ann"]
    hub.stop()


async def test_a_lowered_ttl_forgets_a_later_change_first(db):
    clock = Clock()
    hub, typing, _ = captured(db, clock=clock)
    typing.set("room", "ann")
    typing.ttl = 1
    typing.set("room", "bob")
    clock.now += 1
    typing.set("room", "cy")
    assert typing._sent == {"room": {"ann": 1, "cy": 3}}
    hub.stop()


async def test_a_forked_process_does_not_take_its_parents_changes_for_its_own(db):
    hub, typing, _ = captured(db)
    parent = typing._origin
    with mock.patch("pg_sse._ephemeral.os.getpid", return_value=os.getpid() + 1):
        typing.set("room", "ann", on=False)  # the child's change 1, on its way
        assert typing._origin != parent
        message = {"v": 1, "e": "typing", "k": "room", "w": "ann", "on": True, "ttl": 8, "o": parent, "n": 1}
        hub._receive_ephemeral(json.dumps(message))  # the parent's change 1: older than the child's
        assert typing.live("room") == []
    hub.stop()


async def test_changes_sent_before_the_listener_listens_are_not_waited_for(db):
    hub, typing, _ = captured(db)
    typing.set("room", "ann")
    assert typing._sent == {"room": {"ann": 1}}
    async with hub.lifespan():
        await until(lambda: typing._sent == {})


async def test_shared_traffic_is_not_a_notification_in_the_stats(db):
    a, b, typing_a, typing_b, _ = two_hubs(db)
    async with AsyncExitStack() as stack:
        await running(stack, a, b)
        typing_a.set("room", "ann")
        await until(lambda: typing_b.live("room") == ["ann"])
        assert b.stats().last_notification_at is None


async def test_a_hub_with_a_longer_ttl_sends_one_renewal_to_stretch_the_others_not_one_per_renewal(db):
    clock = Clock()
    notify = Notify(db.dsn)
    a, b = make_hub(db, publish=notify), make_hub(db, publish=notify)
    typing_a = a.ephemeral("typing", ttl=8, shared=True)
    typing_b = b.ephemeral("typing", ttl=60, shared=True, clock=clock)
    async with AsyncExitStack() as stack:
        await running(stack, a, b)
        typing_a.set("room", "ann")
        await until(lambda: typing_b.live("room") == ["ann"])
        for _ in range(5):
            clock.now += 1
            typing_b.set("room", "ann")
        await until(lambda: len(notify.sent) == 2)
        await asyncio.sleep(0.1)
        assert [m["ttl"] for m in notify.sent] == [8, 60]


async def test_a_renewal_that_is_not_sent_builds_no_payload(db):
    clock = Clock()
    hub = make_hub(db, publish=Notify(db.dsn))
    typing = hub.ephemeral("typing", ttl=8, shared=True, clock=clock)
    with mock.patch.object(typing, "_message", wraps=typing._message) as message:
        typing.set("room", "ann")
        clock.now += 1
        typing.set("room", "ann")
        assert message.call_count == 1
    hub.stop()


async def test_a_change_reaches_only_its_keys_streams(db):
    a, b, typing_a, _, _ = two_hubs(db)
    async with AsyncExitStack() as stack:
        await running(stack, a, b)
        room = b.subscribe("room")
        other = b.subscribe("other")
        everything = b.subscribe(None)
        typing_a.set("room", "ann")
        assert await woken(room, 5)
        assert not await woken(other)
        assert not await woken(everything)  # all_keys=False, as for a local change


async def test_a_change_too_large_for_a_notify_raises_and_changes_nothing(db):
    a, _, typing_a, _, notify = two_hubs(db)
    with pytest.raises(ValueError, match="7900"):
        typing_a.set("room", "x" * 7900)
    with pytest.raises(ValueError, match="7900"):
        typing_a.clear("é" * 4000)  # counted in bytes, not characters
    assert typing_a.live("room") == []
    assert a._publisher is None
    assert notify.sent == []
    typing_a.set("x" * 1300, "ann")  # long enough to be measured, short enough to send
    await until(lambda: len(notify.sent) == 1)
    a.stop()


async def test_a_shared_who_must_be_text_or_a_number(db):
    a, _, typing_a, _, notify = two_hubs(db)
    with pytest.raises(TypeError, match="who must be a str or an int"):
        typing_a.set("room", ("ann",))
    with pytest.raises(TypeError, match="who must be a str or an int, got NoneType"):
        typing_a.set("room", None, on=False)  # not a clear of everyone
    assert notify.sent == []
    typing_a.set("room", 7)
    assert typing_a.live("room") == [7]
    a.stop()


@pytest.mark.parametrize(
    ("payload", "why"),
    [
        ("not json", "not a JSON object"),
        ("[1]", "not a JSON object"),
        ('{"v":2,"e":"typing","k":"room","w":"ann","on":true,"ttl":8}', "version 2"),
        ('{"v":true,"e":"typing","k":"room","w":"ann","on":true,"ttl":8}', "version True"),
        ('{"v":1,"e":"drawing","k":"room","w":"ann","on":true,"ttl":8}', "no shared Ephemeral named 'drawing'"),
        ('{"v":1,"e":["typing"],"k":"room","w":"ann","on":true,"ttl":8}', "no shared Ephemeral named"),
        ('{"v":1,"e":"typing","k":1,"w":"ann","on":true,"ttl":8}', "malformed"),
        ('{"v":1,"e":"typing","k":"room","w":"ann","on":1,"ttl":8}', "malformed"),
        ('{"v":1,"e":"typing","k":"room","w":null,"on":true,"ttl":8}', "malformed"),
        ('{"v":1,"e":"typing","k":"room","w":false,"on":false,"ttl":8}', "malformed"),
        ('{"v":1,"e":"typing","k":"room","w":"ann","on":true,"ttl":-1}', "malformed"),
        ('{"v":1,"e":"typing","k":"room","w":"ann","on":true}', "malformed"),
    ],
)
async def test_a_message_that_cannot_be_applied_is_counted_and_logged_not_raised(db, caplog, payload, why):
    a, b, typing_a, typing_b, _ = two_hubs(db)
    async with AsyncExitStack() as stack:
        await running(stack, a, b)
        with caplog.at_level(logging.WARNING, logger="pg_sse"):
            execute(db.dsn, "SELECT pg_notify(%s, %s)", (b.ephemeral_channel, payload))
            await until(lambda: b.stats().ephemeral_rejected_total == 1)
        assert why in caplog.text
        assert typing_b.live("room") == []
        typing_a.set("room", "ann")  # the listener is still there
        await until(lambda: typing_b.live("room") == ["ann"])


async def test_an_unshared_ephemeral_sends_nothing_and_receives_nothing(db):
    notify = Notify(db.dsn)
    a, b = make_hub(db, publish=notify), make_hub(db, publish=notify)
    typing_a = a.ephemeral("typing", ttl=8)
    typing_b = b.ephemeral("typing", ttl=8)
    async with AsyncExitStack() as stack:
        await running(stack, a, b)
        typing_a.set("room", "ann")
        typing_a.clear("room")
        message = {"v": 1, "e": "typing", "k": "room", "w": "bob", "on": True, "ttl": 8}
        execute(db.dsn, "SELECT pg_notify(%s, %s)", (b.ephemeral_channel, json.dumps(message)))
        execute(db.dsn, "SELECT pg_notify(%s, 'room')", (db.channel,))
        await until(lambda: b.stats().last_notification_at is not None)  # the later NOTIFY arrived, the earlier not
        assert typing_b.live("room") == []
        assert b.stats().ephemeral_rejected_total == 0  # not listening on that channel at all
        assert notify.sent == []
        assert a._publisher is None


async def test_a_shared_ephemeral_made_while_the_hub_runs_starts_receiving(db):
    notify = Notify(db.dsn)
    a, b = make_hub(db, publish=notify), make_hub(db, publish=notify, listen_poll_seconds=0.05)
    typing_a = a.ephemeral("typing", ttl=8, shared=True)
    async with AsyncExitStack() as stack:
        await running(stack, a, b)
        typing_b = b.ephemeral("typing", ttl=8, shared=True)
        await asyncio.sleep(0.3)  # the listener's next turn
        typing_a.set("room", "ann")
        await until(lambda: typing_b.live("room") == ["ann"])


async def test_a_publish_that_raises_is_logged_and_counted_and_the_change_still_holds(db, caplog):
    def publish(channel: str, payload: str) -> None:
        raise ConnectionError("database gone")

    hub = make_hub(db, publish=publish)
    typing = hub.ephemeral("typing", ttl=8, shared=True)
    async with hub.lifespan():
        subscription = hub.subscribe("room")
        with caplog.at_level(logging.WARNING, logger="pg_sse"):
            assert typing.set("room", "ann") is True
            assert typing.clear("room", "bob") is False
            await until(lambda: hub.stats().ephemeral_publish_errors_total == 2)
        assert await subscription.wait(1) == [("wake", "room", "typing")]
        assert typing.live("room") == ["ann"]
        assert "publish raised" in caplog.text
        assert "database gone" in caplog.text


async def test_a_change_that_finds_too_many_waiting_is_counted_and_dropped(db, monkeypatch):
    released = threading.Event()
    sent: list[str] = []

    def publish(channel: str, payload: str) -> None:
        released.wait(5)
        sent.append(payload)

    monkeypatch.setattr(hub_module, "_OUTBOX_SIZE", 1)
    hub = make_hub(db, publish=publish)
    typing = hub.ephemeral("typing", ttl=8, shared=True)
    typing.set("room", "ann")  # taken by the publisher, which waits
    await until(lambda: hub._outbox.empty())
    typing.set("room", "bob")  # waits in the queue
    typing.set("room", "cy")  # no room
    assert hub.stats().ephemeral_publish_errors_total == 1
    released.set()
    await until(lambda: len(sent) == 2)
    assert [json.loads(p)["w"] for p in sent] == ["ann", "bob"]
    assert typing.live("room") == ["ann", "bob", "cy"]
    assert typing._sent == {"room": {"ann": 1, "bob": 2}}  # cy's change will not come back
    hub.stop()


async def test_reset_forgets_received_signals(db):
    a, b, typing_a, typing_b, _ = two_hubs(db)
    async with AsyncExitStack() as stack:
        await running(stack, a, b)
        typing_a.set("room", "ann")
        await until(lambda: typing_b.live("room") == ["ann"])
        typing_b.reset()
        assert typing_b.live("room") == []


async def test_leaving_lifespan_sends_what_is_waiting_and_ends_the_publisher(db):
    notify = Notify(db.dsn)

    def slow(channel: str, payload: str) -> None:
        threading.Event().wait(0.05)
        notify(channel, payload)

    hub = make_hub(db, publish=slow)
    typing = hub.ephemeral("typing", ttl=8, shared=True)
    async with hub.lifespan():
        for who in ("ann", "bob", "cy"):
            typing.clear("room", who)  # users leaving as the app shuts down
        publisher = hub._publisher
    assert [m["w"] for m in notify.sent] == ["ann", "bob", "cy"]
    assert not publisher.is_alive()
    typing.set("room", "dee")  # a later change starts another
    await until(lambda: len(notify.sent) == 4)
    hub.stop()


class Fatal(BaseException):
    pass


@pytest.mark.filterwarnings("ignore::pytest.PytestUnhandledThreadExceptionWarning")
async def test_a_publisher_that_died_is_replaced(db):
    sent: list[str] = []

    def publish(channel: str, payload: str) -> None:
        if not sent:
            sent.append("died")
            raise Fatal  # not an Exception: it ends the thread
        sent.append(json.loads(payload)["w"])

    hub = make_hub(db, publish=publish)
    typing = hub.ephemeral("typing", ttl=8, shared=True)
    typing.set("room", "ann")
    await until(lambda: hub._publisher is not None and not hub._publisher.is_alive())
    typing.set("room", "bob")
    await until(lambda: sent == ["died", "bob"])
    hub.stop()


def test_shared_needs_a_publish_function(db):
    with pytest.raises(ValueError, match=r"needs Hub\(publish=...\)"):
        make_hub(db).ephemeral("typing", ttl=8, shared=True)


def test_two_shared_instances_on_one_hub_need_their_own_names(db):
    hub = make_hub(db, publish=Notify(db.dsn))
    hub.ephemeral("typing", ttl=8, shared=True)
    hub.ephemeral("typing", ttl=8)  # not shared: nothing to tell apart
    with pytest.raises(ValueError, match="already has a shared Ephemeral named 'typing'"):
        hub.ephemeral("typing", ttl=8, shared=True)


def test_shared_needs_room_for_the_suffix_in_the_channel_name(db):
    from pg_sse import Hub

    hub = Hub(db.dsn, "c" * 54, handle_signals=False, publish=Notify(db.dsn))
    with pytest.raises(ValueError, match="at most 53 bytes"):
        hub.ephemeral("typing", ttl=8, shared=True)
    assert Hub(db.dsn, "c" * 53, handle_signals=False, publish=Notify(db.dsn)).ephemeral("typing", ttl=8, shared=True)


async def _async_publish(channel: str, payload: str) -> None:
    pass


@pytest.mark.parametrize("publish", [_async_publish, "SELECT pg_notify"])
def test_publish_must_be_a_plain_function(db, publish):
    with pytest.raises(TypeError, match="publish must be a plain function"):
        make_hub(db, publish=publish)
