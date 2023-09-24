import ray


# `auto` is passed to allow the head node
# to determine the networking.
# ray.init(address="auto")
ray.init(address="erinys02.fi.muni.cz:6379")


# Functions can be decorated to tell Ray what function
# will be distributed for compute.
# Decorators work perfectly for simple functions.
@ray.remote
def f(x):
    y = 0
    for i in range(x):
        y += 1
    return y * y


# Manual data processing is done to collect results.
futures = [f.remote(i) for i in range(20)]
results = ray.get(futures)
print(results)