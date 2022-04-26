import time
from multiprocessing import Process, Array, Queue, Value, cpu_count

# Naive method for introducing parallelism with shared memory
# we add all the points that we need to process in a large queue
# all the processes synchronize before we can move on to the next
# set of points.
def foo(q, points_added, points_done):
    r = q.get()
    points_done.value+=1
    # print(r)
    # do something
    b = [i for i in range(int(1e7))]
    # update if condition (if new point along the coronary)
    if r <= 6:
        q.put(r+1)
        points_added.value+=1
        q.put(r+1)
        points_added.value+=1
    return r

if __name__ == "__main__":
    start_parallel = time.time()
    q = Queue()
    arr = Array("d", range(10), lock=True)
    q.put(1)
    q.put(2)
    points_added = Value("i", 2)
    points_done = Value("i", 0)
    
    while points_done.value != points_added.value:
        # print("In while")
        procs = min(points_added.value-points_done.value, cpu_count())
        # print(procs)
        for _ in range(procs):
            p = Process(target=foo, args=(q, points_added, points_done))
            p.start()
        for _ in range(procs):
            p.join()
        # print(points_done.value - points_added.value)
    time_parallel = time.time() - start_parallel
    print(time_parallel)

    start_serial = time.time()
    for _ in range(points_done.value):
        b = [i for i in range(int(1e7))]
    time_serial = time.time() - start_serial
    print(time_serial)