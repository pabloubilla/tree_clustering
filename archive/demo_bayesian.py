# PROJECT SETUP
# set DRAFT = False for final output; DRAFT = True is faster
# https://bob-carpenter.github.io/stan-getting-started/stan-getting-started.html
DRAFT = False

import itertools
import logging
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings( "ignore", module = "plotnine\..*" )

import cmdstanpy as csp
csp.utils.get_logger().setLevel(logging.ERROR)

import numpy as np
import statistics as stat
import pandas as pd
import plotnine as pn
import patchworklib as pw
import time

class StopWatch:
    def __init__(self):
        self.start()
    def start(self):
        self.start_time = time.time()
    def elapsed(self):
        return time.time() - self.start_time
timer = StopWatch()

N = 100
theta = 0.3
data = {'N': N, 'theta': theta}
model = csp.CmdStanModel(stan_file = 'binomial-rng.stan')
sample = model.sample(data = data, seed = 123, chains = 1,
                      iter_sampling = 10, iter_warmup = 0,
                      show_progress = False, show_console = False,
                      adapt_engaged=False)

y = sample.stan_variable('y')
print("N = ", N, ";  theta = ", theta, ";  y(0:10) =", *y.astype(int))

for N in [10, 100, 1_000, 10_000]:
    data = {'N': N, 'theta': theta}
    sample = model.sample(data = data, seed = 123, chains = 1,
                          iter_sampling = 10, iter_warmup = 0,
                          show_progress = False,
              show_console = False, adapt_engaged=False)
    y = sample.stan_variable('y')
    print("N =", N)
    print("  y: ", *y.astype(int))
    print("  est. theta: ", *(y / N))


np.random.seed(123)
ts = []
ps = []
theta = 0.3
M = 100 if DRAFT else 100_000
for N in [10, 100, 1_000]:
    data = {'N': N, 'theta': theta}
    sample = model.sample(data = data, seed = 123, chains = 1,
       iter_sampling = M, iter_warmup = 0, show_progress = False,
       show_console = False, adapt_engaged=False)
    y = sample.stan_variable('y')
    theta_hat = y / N
    ps.extend(theta_hat)
    ts.extend(itertools.repeat(N, M))
xlabel = 'estimated Pr[success]'    
df = pd.DataFrame({xlabel: ps, 'trials': ts})
print(pn.ggplot(df, pn.aes(x = xlabel))
  + pn.geom_histogram(binwidth=0.01, color='white')
  + pn.facet_grid('. ~ trials')
  + pn.scales.scale_x_continuous(limits = [0, 1], breaks = [0, 1/4, 1/2, 3/4, 1],
      labels = ["0", "1/4", "1/2", "3/4", "1"], expand=[0, 0])
  + pn.scales.scale_y_continuous(expand=[0, 0, 0.05, 0])
  + pn.theme(aspect_ratio = 1, panel_spacing = 0.15,
             strip_text = pn.element_text(size = 6),
             strip_background = pn.element_rect(height=0.08,
                                fill = "lightgray")))