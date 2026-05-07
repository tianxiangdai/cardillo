from collections import OrderedDict


class MyLRUCache:
    def __init__(self, maxsize=1024):
        self.cache = OrderedDict()
        self.maxsize = maxsize

    def __getitem__(self, key):
        cache = self.cache
        if key in cache:
            cache.move_to_end(key)  # mark as recently used
            return cache[key]
        return None

    def __setitem__(self, key, value):
        cache = self.cache
        if key in cache:
            cache.move_to_end(key)
        cache[key] = value

        if len(cache) > self.maxsize:
            cache.popitem(last=False)  # delete least recently used item
