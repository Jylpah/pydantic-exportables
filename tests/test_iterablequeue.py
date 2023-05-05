import sys
import pytest # type: ignore
from pathlib import Path
from queue import Queue
from asyncio.queues import QueueEmpty, QueueFull
from asyncio import Task, create_task, sleep, gather, timeout, TimeoutError
from random import random

sys.path.insert(0, str(Path(__file__).parent.parent.resolve() / 'src'))

from pyutils import IterableQueue, QueueDone

QSIZE 	: int = 10
N 		: int = 100   # N >> QSIZE
THREADS : int = 4
# N : int = int(1e10)

@pytest.fixture
def test_interablequeue_int() -> IterableQueue[int]:
	return IterableQueue[int](maxsize=QSIZE)


async def _producer_int(Q: IterableQueue[int], 
						n : int, 
						finish: bool = False,
						wait: float = 0):
	await Q.add_producer(N=1)
	for i in range(n):
		await sleep(wait*random())
		await Q.put(i)
	if finish:
		await Q.finish()


async def _consumer_int(Q: IterableQueue[int], 
						n : int = -1,
						wait: float = 0):
	try:
		while n != 0:		
			_ = await Q.get()
			await sleep(wait*random())
			Q.task_done()
			n -= 1
	except QueueDone:
		pass


# 
@pytest.mark.timeout(10)
@pytest.mark.asyncio
async def test_1_put_get_async(test_interablequeue_int: IterableQueue[int]):
	"""Test: put(), get(), join(), qsize(), empty() == True"""
	Q = test_interablequeue_int	
	try:
		async with timeout(5):
			await _producer_int(Q, QSIZE-1, finish=True)
	except TimeoutError:		
		assert False, "IterableQueue got stuck"
	assert Q.qsize() == QSIZE-1, f"qsize() returned {Q.qsize()}, should be {QSIZE-1}"
	try:
		await Q.put(1)
		assert False, "Queue is done and put() should raise an exception"
	except QueueDone:
		pass # Queue is done and put() should raise an exception
	
	consumer : Task = create_task(_consumer_int(Q))
	try:
		async with timeout(5):
			await Q.join()
		await Q.get()
		assert False, "Queue is done and put() should raise an exception"
	except TimeoutError:
		assert False, "IterableQueue.join() took too long"
	except QueueDone:
		pass # should be raised
	assert Q.qsize() == 0, "queue not empty"
	assert Q.empty(), "queue not empty"
	consumer.cancel()


@pytest.mark.timeout(10)
@pytest.mark.asyncio
async def test_2_put_get_nowait(test_interablequeue_int: IterableQueue[int]):
	Q = test_interablequeue_int
	producer : Task = create_task(_producer_int(Q, N))
	await sleep(1) 
	# In theory this could fail without a real error 
	# if QSIZE is huge and/or system is slow
	assert Q.qsize() == Q.maxsize, "Queue was supposed to be at maxsize"
	assert Q.full(), "Queue should be full"
	assert not Q.empty(), "Queue should not be empty"
	
	try:
		Q.put_nowait(1)
		assert False, "Queue was supposed to be full, but was not"
	except QueueFull:
		pass # OK, Queue was supposed to be full
	
	try:
		while True:
			_ = Q.get_nowait()
			Q.task_done()
			await sleep(0.01)
	except QueueEmpty:
		assert Q.qsize() == 0, "Queue size should be zero"
	
	try:
		async with timeout(5):
			await Q.finish()
			await Q.join()
	except TimeoutError:
		assert False, "Queue.join() took longer than it should"
	assert Q.qsize() == 0, "queue size is > 0 even it should be empty"
	assert Q.empty(), "queue not empty()"
	producer.cancel()
	
	

@pytest.mark.timeout(10)
@pytest.mark.asyncio
async def test_3_multiple_producers(test_interablequeue_int: IterableQueue[int]):
	Q = test_interablequeue_int	
	workers : list[Task] = list()
	for _ in range(THREADS):
		workers.append(create_task(_producer_int(Q, N, finish=True, wait=0.05)))
	try:
		async with timeout(10):
			async for _ in Q:
				pass
	except TimeoutError:
		assert False, "IterableQueue.join() took too long"
	except QueueDone:
		pass # Queue is done

	assert Q.qsize() == 0, f"queue size is {Q.qsize()} even it should be empty"
	assert Q.empty(), "queue not empty"
	for w in workers:
		w.cancel()


@pytest.mark.timeout(10)
@pytest.mark.asyncio
async def test_4_multiple_producers_consumers(test_interablequeue_int: IterableQueue[int]):
	Q = test_interablequeue_int
	producers : list[Task] = list()
	consumers : list[Task] = list()
	
	for _ in range(THREADS):
		producers.append(create_task(_producer_int(Q, N, finish=False, wait=0.05)))
		consumers.append(create_task(_consumer_int(Q, 2*N, wait=0.06)))
	try:
		async with timeout(10):
			await gather(*producers)
			await Q.shutdown()
			await Q.join()
			assert not Q.has_wip, "Queue should not have any items WIP"
	except TimeoutError:
		assert False, "IterableQueue.join() took too long"
	assert Q.count == THREADS*N, f"count returned wrong value {Q.count}, should be {THREADS*N}"
	assert Q.qsize() == 0, "queue size is > 0 even it should be empty"
	assert Q.empty(), "queue not empty"
	for p in consumers:
		p.cancel()
