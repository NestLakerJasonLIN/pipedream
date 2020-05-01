# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import threading
import time

"""
Implementation of a thread-safe queue with one producer and one consumer.
"""


class Queue:
    def __init__(self):
        self.queue = []
        self.cv = threading.Condition()

    def add(self, tensor):
        self.cv.acquire()
        self.start_time = time.time()
        self.queue.append(tensor)
        self.cv.notify()
        self.cv.release()

    def remove(self):
        self.cv.acquire()
        while len(self.queue) == 0:
            self.cv.wait()
        tensor = self.queue.pop(0)
        self.end_time = time.time()
        self.cv.release()
        return tensor

    def get_waittime(self):
        try:
            return (self.end_time - self.start_time) * 1000.0
        except BaseException:
            return -1
