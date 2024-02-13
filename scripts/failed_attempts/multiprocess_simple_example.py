from multiprocess import Process, Queue


def f(q, transmit_queue):

    q.put("hello world")
    with open("./temp/stop.txt", "w") as f:
        f.write("stop")
        f.write(f"queue size {transmit_queue.qsize()} \n")


if __name__ == "__main__":
    q = Queue()
    transmit_queue = Queue()
    transmit_queue.put("hello world")
    p = Process(target=f, args=[q, transmit_queue])
    p.start()
    print(q.get())
    p.join()
