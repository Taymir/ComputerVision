#coding=utf-8
import random
from matplotlib import pyplot as plt
from collections import Counter
import math



def random_kid():
    return random.choice(['boy', 'girl'])

def test_distribution():
    both_girls = 0
    older_girl = 0
    either_girl = 0

    #random.seed(0)

    for _ in range(10000):
        younger = random_kid()
        older = random_kid()

        if older == 'girl':
            older_girl += 1
        if older == 'girl' and younger == 'girl':
            both_girls += 1
        if older == 'girl' or younger == 'girl':
            either_girl += 1

    print("P(обе | старшая): ", both_girls / older_girl)
    print("P(обе | любая): ", both_girls / either_girl)


#test_distribution()

def uniform_pdf(x):
    return 1 if x >= 0 and x < 1 else 0

def test_uniform_pdf():
    xs = [x / 10.0 for x in range(-10, 20)]
    plt.plot(xs, [uniform_pdf(x) for x in xs], '-')
    plt.show()
#test_uniform_pdf()

def uniform_cdf(x):
    if x<0:
        return 0
    elif x<1:
        return x
    else:
        return 1

def test_uniform_cdf():
    xs = [x / 10.0 for x in range(-10, 20)]
    plt.plot(xs, [uniform_cdf(x) for x in xs], '-')
    plt.show()
#test_uniform_cdf()

def normal_pdf(x, mu=0, sigma=1):
    sqrt_two_pi = math.sqrt(2 * math.pi)
    return  (math.exp(-(x-mu) ** 2 / 2 / sigma **2) / (sqrt_two_pi * sigma))

def test_normal_pdf():
    xs = [x / 10.0 for x in range(-50, 50)]
    plt.plot(xs, [normal_pdf(x, sigma=1) for x in xs], '-', label='mu=0, sigma=1')
    plt.plot(xs, [normal_pdf(x, sigma=2) for x in xs], '--', label='mu=0, sigma=2')
    plt.plot(xs, [normal_pdf(x, sigma=0.5) for x in xs], ':', label='mu=0, sigma=0.5')
    plt.plot(xs, [normal_pdf(x, mu=-1) for x in xs], '-.', label='mu=-1, sigma=1')
    plt.legend()
    plt.title("Some normal distributions")
    plt.show()
#test_normal_pdf()

def normal_cdf(x, mu=0, sigma=1):
    return (1+math.erf((x - mu) / math.sqrt(2) / sigma)) / 2

def test_normal_cdf():
    xs = [x / 10.0 for x in range(-50, 50)]
    plt.plot(xs, [normal_cdf(x, sigma=1) for x in xs], '-', label='mu=0, sigma=1')
    plt.plot(xs, [normal_cdf(x, sigma=2) for x in xs], '--', label='mu=0, sigma=2')
    plt.plot(xs, [normal_cdf(x, sigma=0.5) for x in xs], ':', label='mu=0, sigma=0.5')
    plt.plot(xs, [normal_cdf(x, mu=-1) for x in xs], '-.', label='mu=-1, sigma=1')
    plt.legend(loc=4)
    plt.title("Some normal distributions")
    plt.show()
#test_normal_cdf()

def bernoulli_trial(p):
    return 1 if random.random() < p else 0

def binomial(n,p):
    return sum(bernoulli_trial(p) for _ in range(n))

def make_hist(p, n, num_points):
    data = [binomial(n,p) for _ in range(num_points)]
    #print (data)
    histogram = Counter(data)
    plt.bar([x - 0.4 for x in histogram.keys()],
            [v/num_points for v in histogram.values()],
            0.8,
            color='0.75')
    mu = p*n
    sigma = math.sqrt(n* p * (1-p))

    xs = range(min(data), max(data) + 1)
    ys = [normal_cdf(i + 0.5, mu, sigma) - normal_cdf(i-0.5, mu, sigma) for i in xs]
    plt.plot(xs, ys)
    plt.title ("Binomial distribution and its normal approximation")
    plt.show()
#make_hist(0.75, 100, 10000)

def inverse_normal_cdf(p, mu=0, sigma=1, tolerance=0.00001):
    if mu != 0 or sigma != 1:
        return mu + sigma * inverse_normal_cdf(p, tolerance=tolerance)

    low_z, low_p = -10.0, 0
    hi_z, hi_p = 10.0, 1

    while hi_z - low_z > tolerance:
        mid_z = (low_z + hi_z) / 2
        mid_p = normal_cdf(mid_z)
        if mid_p < p:
            low_z, low_p = mid_z, mid_p
        elif mid_p > p:
            hi_z, hi_p = mid_z, mid_p
        else:
            break

    return mid_z

def normal_approximation_to_binomial (n,p):
    mu = p*n
    sigma = math.sqrt(p * (1-p) * n)
    return mu, sigma

normal_probability_below = normal_cdf

def normal_probability_above(lo, mu=0, sigma=1):
    return 1-normal_cdf(lo,mu, sigma)

def normal_probability_between(lo, hi, mu=0, sigma=1):
    return normal_cdf(hi, mu, sigma) - normal_cdf(lo, mu, sigma)

def normal_probability_outside(lo, hi, mu=0, sigma=1):
    return 1 - normal_probability_between(lo, hi, mu, sigma)

def normal_upper_bound(probability, mu=0, sigma=1):
    return inverse_normal_cdf(probability, mu, sigma)

def normal_lower_bound(probability, mu=0, sigma=1):
    return inverse_normal_cdf(1 - probability, mu, sigma)

def normal_two_sided_bounds(probability, mu=0, sigma=1):
    tail_probability = (1 - probability) / 2
    upper_bound = normal_lower_bound(tail_probability, mu, sigma)
    lower_bound = normal_upper_bound(tail_probability, mu, sigma)

    return lower_bound, upper_bound

mu_0, sigma_0 = normal_approximation_to_binomial(1000, 0.5)
print(normal_two_sided_bounds(0.95, mu_0, sigma_0))
lo, hi = normal_two_sided_bounds(0.95, mu_0, sigma_0)
mu_1, sigma_1 = normal_approximation_to_binomial(1000, 0.55)
type_2_probability = normal_probability_between(lo, hi, mu_1, sigma_1)
power = 1 - type_2_probability
print(type_2_probability)

hi = normal_upper_bound(0.95, mu_0, sigma_0)
print(hi)
type_2_probability = normal_probability_below(hi, mu_1, sigma_1)
power = 1 - type_2_probability
print (power)
