from multiprocessing import Queue, cpu_count
import multiprocessing
import time
from concurrent.futures import ProcessPoolExecutor, wait, ALL_COMPLETED

WAIT_SLEEP = 0.1 # second to adjust based on our application

class BufferedIter(object):
    def __init__(self):
        pass
        # self.queue = input_queue
        # print(self.queue.get())

    def nextN(self, input_queue, n):
        vals = []
        for _ in range(n):
            if input_queue.empty():
                continue
            vals.append(input_queue.get(timeout=0))
        return vals

def queue_processor(input_queue, task, num_workers):
    futures = dict()
    buffer = BufferedIter()

    with ProcessPoolExecutor(num_workers) as executor:
        while True:
            idle_workers = num_workers - len(futures)
            # print(idle_workers)
            points = buffer.nextN(input_queue, idle_workers)
            for point in points:
                futures[executor.submit(task, input_queue, point)] = point 
            done, _ = wait(futures, timeout=WAIT_SLEEP, return_when=ALL_COMPLETED)

            for f in done:
                data = futures[f]
                try:
                    ret = f.result(timeout=0)
                    print(ret)
                except Exception as exc:
                    print(f'future encountered an exception {data, exc}')
                del futures[f]

            if input_queue.empty() and len(futures)==0:
                break
            
def task(input_queue, r):
    """ Task to perform - let it sleep for a bit."""
    # print("doing task")
    # print(r)
    # time.sleep(1)
    b = [i for i in range(int(1e7))]
    print(r["name"])
    # update if condition (if new point along the coronary)
    # if r <= 6:
    #     input_queue.put(r+1)
    #     input_queue.put(r+1)
    # return r * r
    return r

if __name__ == "__main__":
    start = time.time()
    m = multiprocessing.Manager()
    input_queue = m.Queue()
    # input_queue = Queue()
    input_queue.put({"name":"Hi", "radius":0, "location":[1, 2, 3]})
    input_queue.put({"name":"Hello", "radius":5, "location":[6, 7, 8]})
    # input_queue.put(2)
    num_workers = cpu_count()
    queue_processor(input_queue, task, num_workers)
    print(time.time() - start)

    