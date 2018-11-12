import sys
sys.path.append("./")
import time
from bayes_opt import BayesianOptimization
from bayes_opt.util import UtilityFunction

import asyncio
import threading

try:
    import json
    import tornado.ioloop
    import tornado.httpserver
    from tornado.web import RequestHandler
    import requests
except ImportError:
    raise ImportError(
        "In order to run this example you must have the libraries: " +
        "`tornado` and `requests` installed."
    )


def black_box_function(x, y):
    """Function with unknown internals we wish to maximize.

    This is just serving as an example, however, for all intents and
    purposes think of the internals of this function, i.e.: the process
    which generates its outputs values, as unknown.
    """
    time.sleep(1)
    return -x ** 2 - (y - 1) ** 2 + 1


class BayesianOptimizationHandler(RequestHandler):
    """Basic functionality for NLP handlers."""
    _bo = BayesianOptimization(
        f=black_box_function,
        pbounds={"x": (-4, 4), "y": (-3, 3)}
    )
    _uf = UtilityFunction(kind="ucb", kappa=3, xi=1)

    def post(self):
        """Deal with incoming requests."""
        body = tornado.escape.json_decode(self.request.body)

        if body:
            self._bo.probe(body["params"], lazy=False)

        suggested_params = self._bo.suggest(self._uf)

        output = {
            "target": self._bo.res[-1]["target"] if self._bo.res else None,
            "suggested_params": suggested_params,
        }
        self.write(json.dumps(output))


def run_optimization_app():
    asyncio.set_event_loop(asyncio.new_event_loop())
    handlers = [
        (r"/bayesian_optimization", BayesianOptimizationHandler),
    ]
    server = tornado.httpserver.HTTPServer(
        tornado.web.Application(handlers)
    )
    server.listen(9009)
    tornado.ioloop.IOLoop.instance().start()


def run_optimizer():
    x = {}
    global optimizers_names
    name = optimizers_names.pop()

    max_target = None

    for _ in range(10):
        status = name + " wants to probe: {}.\n".format(x)

        resp = requests.post(
            url="http://localhost:9009/bayesian_optimization",
            json=x
        ).json()
        x = {"params": resp["suggested_params"]}

        if max_target is None or resp["target"] > max_target:
            max_target = resp["target"]

        status += name + " got {} as target.\n".format(resp["target"])
        status += name + " will to probe next: {}.\n".format(x)
        print(status, end="\n")

    global results
    results.append((name, max_target))


if __name__ == "__main__":
    ioloop = tornado.ioloop.IOLoop.instance()
    optimizers_names = [
        "optimizer 1",
        "optimizer 2",
        "optimizer 3",
    ]

    app_thread = threading.Thread(target=run_optimization_app)
    app_thread.daemon = True
    app_thread.start()

    targets = (
        run_optimizer,
        run_optimizer,
        run_optimizer
    )
    optimizer_threads = []
    for target in targets:
        optimizer_threads.append(threading.Thread(target=target))
        optimizer_threads[-1].daemon = True
        optimizer_threads[-1].start()

    results = []
    for optimizer_thread in optimizer_threads:
        optimizer_thread.join()

    for result in results:
        print(result[0], "found a maximum value of: {}".format(result[1]))

    ioloop.stop()
