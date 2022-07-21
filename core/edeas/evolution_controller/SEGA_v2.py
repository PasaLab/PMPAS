# -*- coding: utf-8 -*-
import time
import json
import warnings
from os import path
import os.path as osp
from sys import path as paths

import geatpy as ea
import numpy as np

paths.append(path.split(path.split(path.realpath(__file__))[0])[0])


class Algorithm:
    """
    The Algorithm class is used to save the running parameters of the evolution based algorithm.
    attributes:
        name: <str> algorithm name
        search_space: <class> Search Space
        population: <class> Population
        MAXGEN: <int> maximum iterations
        currentGen: <int> current iterations
        MAXTIME: <float> maximum running time(seconds)
        timeSlot: <float> time slot(seconds)
        passTime: <float> time have already passed(seconds)
        MAXEVALS: <int> maximum evaluation times
        evalsNum: <int> current evalution times
        MAXSIZE: <int> best individual maximum number
        logTras: <int> cycle used to logging when running, 0 means not logging, 10 means logging every 10 iterations.
        log: <Dict> logging record which contains two basic key 'gen' and 'eval', 'gen' record the iterations
            of the population, 'eval' record the evalution times. log will be set to None of the logTras==0.
        verbose: <boolean> Used to controller whether print logging information in the input and output stream.
    method:
        __init__()       : Initialization function.
        initialization() : Initialization some dynamic args.
        run()            : Running function.
        logging()        : Logging function.
        stat()           : Statistics of the running process.
        terminated()     : Controlling the running process.
        finishing ()     : Work when the runnning process finished.
        check()          : Check objV.
    """

    def __init__(self):
        self.name = 'Algorithm'
        self.problem = None
        self.population = None
        self.MAXGEN = None
        self.currentGen = None
        self.MAXTIME = None
        self.timeSlot = None
        self.passTime = None
        self.MAXEVALS = None
        self.evalsNum = None
        self.MAXSIZE = None
        self.logTras = None
        self.log = None
        self.verbose = None

    def initialization(self):
        pass

    def run(self, pop):
        pass

    def logging(self, pop):
        pass

    def stat(self, pop):
        pass

    def terminated(self, pop):
        pass

    def finishing(self, pop):
        pass

    def check(self, pop):
        if np.any(np.isnan(pop.ObjV)):
            warnings.warn(
                "Warning: Some elements of ObjV are NAN, please check the calculation of ObjV.",
                RuntimeWarning)
        elif np.any(np.isinf(pop.ObjV)):
            warnings.warn(
                "Warning: Some elements of ObjV are Inf, please check the calculation of ObjV.",
                RuntimeWarning)
        if pop.CV is not None:
            if np.any(np.isnan(pop.CV)):
                warnings.warn(
                    "Warning: Some elements of CV are NAN, please check the calculation of CV.",
                    RuntimeWarning)
            elif np.any(np.isinf(pop.CV)):
                warnings.warn(
                    "Warning: Some elements of CV are Inf, please check the calculation of CV.",
                    RuntimeWarning)

    def call_aimFunc(self, pop):
        pop.Phen = pop.decoding()  # decode
        if self.problem is None:
            raise RuntimeError('error: search_space has not been initialized.')
        self.problem.aimFunc(pop)
        self.evalsNum = self.evalsNum + pop.sizes if self.evalsNum is not None else pop.sizes
        if not isinstance(pop.ObjV, np.ndarray) or pop.ObjV.ndim != 2 or pop.ObjV.shape[0] != pop.sizes or \
                pop.ObjV.shape[1] != self.problem.M:
            raise RuntimeError('error: ObjV is illegal.')
        if pop.CV is not None:
            if not isinstance(pop.CV, np.ndarray) or pop.CV.ndim != 2 or pop.CV.shape[0] != pop.sizes:
                raise RuntimeError('error: CV is illegal.')

    def display(self):
        self.passTime += time.time() - self.timeSlot
        headers = []
        widths = []
        values = []
        for key in self.log.keys():
            if key == 'gen':
                width = max(3, len(str(self.MAXGEN - 1)))
            elif key == 'eval':
                width = 8
            else:
                width = 13
            headers.append(key)
            widths.append(width)
            value = self.log[key][-1] if len(self.log[key]) != 0 else "-"
            if isinstance(value, float):
                values.append("%.5E" % value)
            else:
                values.append(value)
        if len(self.log['gen']) == 1:
            header_regex = '|'.join(['{}'] * len(headers))
            header_str = header_regex.format(*[str(key).center(width) for key, width in zip(headers, widths)])
            print("=" * len(header_str))
            print(header_str)
            print("-" * len(header_str))
        if len(self.log['gen']) != 0:
            value_regex = '|'.join(['{}'] * len(values))
            value_str = value_regex.format(*[str(value).center(width) for value, width in zip(values, widths)])
            print(value_str)
        self.timeSlot = time.time()


class SoeaAlgorithm(Algorithm):

    """
    Class of the Single object evolutionary algorithm.
    attributes:
        trappedValue: <int> Threshold when the algorithm doesn't improved.
        maxTrappedCount: <int> maximum trapped counts.
        drawing: <int>  0 means not drawing, 1 means drawing static graph, 2 and 3 means drawing dynamic graph.
    """
    def __init__(self, problem, population):
        super().__init__()
        self.problem = problem
        self.population = population
        self.trappedValue = 0
        self.maxTrappedCount = 1000
        self.logTras = 1
        self.verbose = True
        self.drawing = 1

        self.BestIndi = None
        self.trace = None
        self.trappedCount = None
        self.ax = None

    def initialization(self):
        self.ax = None
        self.passTime = 0
        self.trappedCount = 0
        self.currentGen = 0
        self.evalsNum = 0
        self.BestIndi = ea.Population(None, None, 0)
        self.log = {'gen': [], 'eval': []} if self.logTras != 0 else None
        self.trace = {'f_best': [], 'f_avg': []}
        # begin timing
        self.timeSlot = time.time()

    def logging(self, pop):

        self.passTime += time.time() - self.timeSlot
        if len(self.log['gen']) == 0:
            self.log['f_opt'] = []
            self.log['f_max'] = []
            self.log['f_avg'] = []
            self.log['f_min'] = []
            self.log['f_std'] = []
        self.log['gen'].append(self.currentGen)
        self.log['eval'].append(self.evalsNum)
        self.log['f_opt'].append(self.BestIndi.ObjV[0][0])
        self.log['f_max'].append(np.max(pop.ObjV))
        self.log['f_avg'].append(np.mean(pop.ObjV))
        self.log['f_min'].append(np.min(pop.ObjV))
        self.log['f_std'].append(np.std(pop.ObjV))
        self.timeSlot = time.time()

    def draw(self, pop, EndFlag=False):
        if not EndFlag:
            self.passTime += time.time() - self.timeSlot

            if self.drawing == 2:
                metric = np.array(self.trace['f_best']).reshape(-1, 1)
                self.ax = ea.soeaplot(metric, Label='Objective Value', saveFlag=False, ax=self.ax, gen=self.currentGen,
                                      gridFlag=False)
            elif self.drawing == 3:
                self.ax = ea.varplot(pop.Phen, Label='decision variables', saveFlag=False, ax=self.ax,
                                     gen=self.currentGen, gridFlag=False)
            self.timeSlot = time.time()
        else:

            if self.drawing != 0:
                metric = np.vstack(
                    [self.trace['f_avg'], self.trace['f_best']]).T
                ea.trcplot(metric, [['population average aimFunc value', 'population best aimFunc value']], xlabels=[['Number of Generation']],
                           ylabels=[['Value']], gridFlags=[[False]])

    def stat(self, pop):
        feasible = np.where(np.all(pop.CV <= 0, 1))[0] if pop.CV is not None else np.arange(pop.sizes)
        if len(feasible) > 0:
            feasiblePop = pop[feasible]
            bestIndi = feasiblePop[np.argmax(feasiblePop.FitnV)]
            if self.BestIndi.sizes == 0:
                self.BestIndi = bestIndi
            else:
                delta = (
                                self.BestIndi.ObjV - bestIndi.ObjV) * self.problem.maxormins if self.problem.maxormins is not None else self.BestIndi.ObjV - bestIndi.ObjV

                self.trappedCount += 1 if np.abs(delta) < self.trappedValue else 0

                if delta > 0:
                    self.BestIndi = bestIndi

            self.trace['f_best'].append(bestIndi.ObjV[0][0])
            self.trace['f_avg'].append(np.mean(feasiblePop.ObjV))
            if self.logTras != 0 and self.currentGen % self.logTras == 0:
                self.logging(feasiblePop)
                if self.verbose:
                    self.display()
            self.draw(feasiblePop)

    def terminated(self, pop):
        self.check(pop)
        self.stat(pop)
        self.passTime += time.time() - self.timeSlot
        self.timeSlot = time.time()
        if (
                self.MAXTIME is not None and self.passTime >= self.MAXTIME) or self.currentGen + 1 >= self.MAXGEN or self.trappedCount >= self.maxTrappedCount:
            return True
        else:
            self.currentGen += 1
            return False

    def finishing(self, pop):
        feasible = np.where(np.all(pop.CV <= 0, 1))[0] if pop.CV is not None else np.arange(pop.sizes)
        if len(feasible) > 0:
            feasiblePop = pop[feasible]
            if self.logTras != 0 and (len(self.log['gen']) == 0 or self.log['gen'][-1] != self.currentGen):
                self.logging(feasiblePop)
                if self.verbose:
                    self.display()
        self.passTime += time.time() - self.timeSlot
        self.draw(pop, EndFlag=True)

        return [self.BestIndi, pop]


class soea_SEGA_templet(ea.SoeaAlgorithm):
    """
    soea_SEGA_templet: <class> Strengthen Elitist GA templet
    algorithm steps:
    1) encoding N individuals
    2) whether stopping? if not, continue
    3) statistics of the population
    4) select N individuals from the population independently
    5) crossover N individuals  independently
    6) mutate N individuals  independently
    7) merge origin population and the crossover population, get 2N individuals
    8) select N individuals from the population of 2N individuals to generate new population
    9) return to step 2)
    larger crossover and mutate probability is recommend to prevent too much duplicate individuals.
    """

    def __init__(self, problem, population, selOperator, recOperator=None, mutOperator=None, encOperator=None,
                 result_path='', stop_by_time=False):
        ea.SoeaAlgorithm.__init__(self, problem, population)
        if population.ChromNum != 1:
            raise RuntimeError('Must passed single DNA type.')
        self.name = 'SEGA'

        self.selFunc = selOperator

        if recOperator is not None:
            self.recOper = recOperator
        if mutOperator is not None:
            self.mutOper = mutOperator

        if encOperator is not None:
            population.Encoding = encOperator

        if population.Encoding == 'P':
            self.recOper = ea.Xovpmx(XOVR=0.7)
            self.mutOper = ea.Mutinv(Pm=0.5)
        else:
            self.recOper = ea.Xovdp(XOVR=0.7)
            if population.Encoding == 'BG':
                self.mutOper = ea.Mutbin(Pm=None)
            elif population.Encoding == 'RI':
                self.mutOper = ea.Mutbga(Pm=1 / self.problem.Dim, MutShrink=0.5, Gradient=20)
            else:
                raise RuntimeError("encoding must be 'BG','RI' or 'P'.'")
        self.result_path = result_path
        self.stop_by_time = stop_by_time

    def run(self, prophetPop=None):

        population = self.population
        NIND = population.sizes
        self.initialization()

        population.initChrom(NIND)
        self.call_aimFunc(population)

        if prophetPop is not None:
            population = (prophetPop + population)[:NIND]
        population.FitnV = ea.scaling(population.ObjV, population.CV, self.problem.maxormins)

        while self.terminated(population) == False:
            # 选择
            offspring = population[ea.selecting(self.selFunc, population.FitnV, NIND)]
            # 进行进化操作
            offspring.Chrom = self.recOper.do(offspring.Chrom)
            offspring.Chrom = self.mutOper.do(offspring.Encoding, offspring.Chrom, offspring.Field)
            self.call_aimFunc(offspring)
            population = population + offspring
            population.FitnV = ea.scaling(population.ObjV, population.CV, self.problem.maxormins)
            # 得到新一代种群
            population = population[ea.selecting('dup', population.FitnV, NIND)]
        return self.finishing(population)

    def draw(self, pop, EndFlag=False):
        if not EndFlag:
            self.passTime += time.time() - self.timeSlot

            if self.drawing == 2:
                metric = np.array(self.trace['f_best']).reshape(-1, 1)
                self.ax = ea.soeaplot(metric, Label='Objective Value', saveFlag=False, ax=self.ax, gen=self.currentGen,
                                      gridFlag=False)
            elif self.drawing == 3:
                self.ax = ea.varplot(pop.Phen, Label='decision variables', saveFlag=False, ax=self.ax,
                                     gen=self.currentGen, gridFlag=False)
            self.timeSlot = time.time()
        else:

            if self.drawing != 0:
                metric = np.vstack(
                    [self.trace['f_avg'], self.trace['f_best']]).T
                ea.trcplot(metric, [['population average aimFunc value', 'population best aimFunc value']], xlabels=[['iterations']],
                           ylabels=[['accuracy']], gridFlags=[[False]], save_path=osp.join(self.result_path, ''))

    def terminated(self, pop):
        self.check(pop)
        self.stat(pop)
        self.passTime += time.time() - self.timeSlot
        self.timeSlot = time.time()

        if self.stop_by_time:
            if self.MAXTIME is not None and self.passTime >= self.MAXTIME:
                return True
            else:
                self.currentGen += 1
                return False
        else:
            if (
                    self.MAXTIME is not None and self.passTime >= self.MAXTIME) or self.currentGen + 1 >= self.MAXGEN or self.trappedCount >= self.maxTrappedCount:
                return True
            else:
                self.currentGen += 1
                return False

    def finishing(self, pop):
        feasible = np.where(np.all(pop.CV <= 0, 1))[0] if pop.CV is not None else np.arange(
            pop.sizes)
        if len(feasible) > 0:
            feasiblePop = pop[feasible]
            if self.logTras != 0 and (
                    len(self.log['gen']) == 0 or self.log['gen'][-1] != self.currentGen):
                self.logging(feasiblePop)
                if self.verbose:
                    self.display()
        self.passTime += time.time() - self.timeSlot
        self.draw(pop, EndFlag=True)

        # todo save experiment results
        with open(osp.join(self.result_path, 'log.json'), 'w', encoding='utf-8') as f:
            json.dump(self.log, f)
        return [self.BestIndi, pop]
