import numpy as np


class Logger:
    def __init__(self):
        self.log_data = dict()

    def log(self, key, value):
        if key not in self.log_data:
            self.log_data[key] = []
        self.log_data[key].append(value)

    def op(self, key, operater, keep=False):
        try:
            if key not in self.log_data:
                print("no key '{}' in logger".format(key))
                return None
            else:
                result = operater(self.log_data[key])
                if not keep:
                    self.log_data[key] = []
                return result
        except:
            print("can not execute operator on data with key '{}'".format(key))
            return None

    def show(self, msg=None, ops=None, map=None, keep=False):
        print('-'*50)
        if msg is not None:
            print(msg)
        if ops is not None:
            if map is None:
                print("require operator mapping")
                return
            for key, op in ops.items():
                if key not in self.log_data:
                    print("no key '{}' in logger".format(key))
                else:
                    print(key, end='')
                    value = self.log_data[key]
                    for name in op:
                        func = map[name]
                        result = func(value)
                        if abs(result) < 1e3 and abs(result) > 1e-2:
                            print("\t {}: {:.2f} ".format(name, result), end='')
                        else:
                            print("\t {}: {:.2e} ".format(name, result), end='')
                    print()
                    if not keep:
                        self.log_data[key] = []
        print('-'*50)


if __name__ == '__main__':
    import torch
    logger = Logger()
    logger.op("wrong key", np.mean)
    logger.log("123", 1)
    ops = {'123': ["mean", "min"]}
    map = {"mean": np.mean}
    logger.show(ops, map)