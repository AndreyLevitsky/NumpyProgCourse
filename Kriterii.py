import itertools
from numpy import prod
from numpy import linspace
from itertools import groupby
from functools import reduce

# node scheme
# {'m': {'f': None, 't': None, 'a': None}, 'w': {'f': None, 't': None, 'a': None}, 'c': {'f': None, 't': None, 'a': None}}

class Table:

    def __init__(self, players, strats):
        self.players = players
        self.stratsInitial = strats
        self.table = []

    def set(self, path, value): # path - {'m': 'f','w': 'f','c': 'f'}; value - (1,2,3)
        elem = self.get(path)
        if not elem:
            elem = {player: {strat: None for strat in self.stratsInitial[player]} for player in self.players}
            self.table.append(elem)
        for num, player in enumerate(path):
            elem[player][path[player]] = value[num]

    def setAllAtOnce(self, data):
        values = iter(data)
        stratsProduct = [p for p in itertools.product(*list(self.stratsInitial.values()))]

        for strat in stratsProduct:
            self.set({p: s for p,s in zip(self.players, strat)}, next(values))

    def get(self, path): # path - {'m': 'f','w': 'f','c': 'f'}
        res = self.table
        for player in path:
            res = list(filter(lambda cell: cell[player][path[player]] != None, res))
        if res: res = res[0]
        return res or None

    def modMaxSearch(self, d):
        maxS, maxV = '', -100
        for strat in d:
            if maxV < d[strat]:
                maxV, maxS = d[strat], strat
        return {maxS: maxV}

    def modFilter(self, p, s, table): # Возвращает все элементы, в которых у игрока player стратегия не None
        return list(filter(lambda elem: elem[p][s] != None, table))

    def GurwicLambda(self, minStrat, maxStrat, ld):
        maxS, maxV = '', -100
        for strat in minStrat.keys():
            temp = minStrat[strat] * (1 - ld) + maxStrat[strat] * ld
            if maxV < temp:
                maxV, maxS = temp, strat
        return {maxS: maxV}

    def allMinMaxStrats(self, mod): # Возвращает макс. или мин. значения всех стратегий для всех игроков
        allStratsMinMax = {player: {strat: None for strat in self.stratsInitial[player]} for player in self.players}

        for player in self.players:
            for strat in self.stratsInitial[player]:
                if mod == 'max':
                    allStratsMinMax[player][strat] = max(
                        list(map(lambda elem: elem[player][strat], self.modFilter(player, strat, self.table))))
                elif mod == 'min':
                    allStratsMinMax[player][strat] = min(
                        list(map(lambda elem: elem[player][strat], self.modFilter(player, strat, self.table))))
        return allStratsMinMax

    def compareSituationsGains(self, firstGains, secondGains):
        compareRes = list(map(lambda couple: couple[0] > couple[1], zip(firstGains, secondGains)))
        if compareRes.count(True) != 0: return True
        else: return False

    def getOnlyGain(self, cell): # возвращает кортеж выигрышей игроков для конкретной ситуации cell
        return [tuple(filter(lambda strat: strat != None, list(cell[player].values())))[0] for player in cell.keys()]

    def fancyCell(self, cell):
        fancy = {}
        for player in self.players:
            for strat in self.stratsInitial[player]:
                if cell[player][strat] != None:
                    fancy.update({player: {strat: cell[player][strat]}})
        return fancy

    def Pareto(self):
        Res = []
        for numi, i in enumerate(self.table):
            for numj, j in enumerate(self.table):
                if numi == numj: continue
                if not self.compareSituationsGains(self.getOnlyGain(i), self.getOnlyGain(j)): break
            else: Res.append(i)

        return [self.fancyCell(cell) for cell in Res]

    def getFloatPlayerStrategySituations(self, player, cell): # возвращает все ситуации, в которых стратегия player не фиксирована, а стратегии остальных как в cell
        res = []
        for p in cell:
            if p == player: continue
            for strat in cell[p]:
                if cell[p][strat] != None:
                    res = self.modFilter(p, strat, res or self.table)
        return res

    def getNotNoneStrat(self, strats):
        return strats[list(filter(lambda strat: strats[strat] != None, strats))[0]]

    def Nesh(self):
        Res = []
        for cell in self.table:
            for num, player in enumerate(self.players):
                if self.getOnlyGain(cell)[num] < max([self.getNotNoneStrat(cell[player]) for cell in self.getFloatPlayerStrategySituations(player, cell)]):
                    break
            else: Res.append(cell)

        return [self.fancyCell(cell) for cell in Res]

    def Sevij(self):
        allMaxStrats = self.allMinMaxStrats('max')

        maxValueStrat = {player: self.modMaxSearch(allMaxStrats[player]) for player in self.players} # {'m': {'f': 8}, 'w': {'t': 8}}

        for player in self.players:
            strategy = list(maxValueStrat[player].keys())[0]
            for strat in self.stratsInitial[player]:
                allMaxStrats[player][strat] = maxValueStrat[player][strategy] - allMaxStrats[player][strat]

        return {player: self.modMaxSearch(allMaxStrats[player]) for player in self.players}

    def Laplas(self):
        temp = {player: {strat: None for strat in self.stratsInitial[player]} for player in self.players}
        for player in self.players:
            for strat in self.stratsInitial[player]:
                stratValues = [elem[player][strat] for elem in self.modFilter(player, strat, self.table)]
                value = sum(stratValues) / len(stratValues)
                temp[player][strat] = value

        return {player: self.modMaxSearch(temp[player]) for player in self.players}

    def allProbability(self, playerOpinions, product): # playerOpinions - мнение игрока player о стратегиях других игроков
        return [playerOpinions[p][product[num]] for num, p in enumerate(playerOpinions.keys())]

    def Bayes(self, opinions): # opinions - {m: {w: {f: None, t: None, a: None}, c: {f: None, t: None, a: None}}, w: {m: {f: None, t: None, a: None}, c: {f: None, t: None, a: None}}, ...}
        temp = {player: {strat: None for strat in self.stratsInitial[player]} for player in self.players}
        for player in self.players:
            for strat in self.stratsInitial[player]:
                playerStratValues = [elem[player][strat] for elem in self.modFilter(player, strat, self.table)] # список значений стратегии strat для игрока player
                stratsProduct = list(itertools.product(*[self.stratsInitial[p] for p in self.stratsInitial if p != player]))


                temp[player][strat] = sum([playerStratValues[num] * prod(self.allProbability(opinions[player], product)) for num, product in enumerate(stratsProduct)])
                # prod (перемножение), так как считаем что принятие игроками стратегий происходит независимо
        return {player: self.modMaxSearch(temp[player]) for player in self.players}

    def deleteStratForPlayer(self, player, strat):
        flag = True
        while flag:
            flag = False
            for cell in self.table:
                if cell[player][strat] != None:
                    self.table.remove(cell)
                    flag = True
        self.stratsInitial[player].remove(strat)

    def dominatedStrat(self):
        flag = True
        while flag:
            flag = False
            for player in self.players:
                for strat in self.stratsInitial[player]:
                    gains = [cell[player][strat] for cell in self.modFilter(player, strat, self.table)]
                    for anotherStrat in self.stratsInitial[player]:
                        if strat == anotherStrat: continue
                        anotherGains = [cell[player][anotherStrat] for cell in self.modFilter(player, anotherStrat, self.table)]
                        if all([x >= y for x,y in zip(gains, anotherGains)]) and True in [x > y for x,y in zip(gains, anotherGains)]:
                            self.deleteStratForPlayer(player, anotherStrat)
                            flag = True
                        if flag: break
                    if flag: break
                if flag: break

    def Gurwic(self, ld):
        return {player: self.GurwicLambda(self.allMinMaxStrats('min')[player], self.allMinMaxStrats('max')[player], ld) for player in self.players}

    def optimistic(self):
        return {player: self.modMaxSearch(self.allMinMaxStrats('max')[player]) for player in self.players}

    def Valdo(self):
        return {player: self.modMaxSearch(self.allMinMaxStrats('min')[player]) for player in self.players}

    def randomGeneric(self):
        return

    def __str__(self):
        for elem in self.table:
            print(elem)
        return ''


tKP = Table(('p1', 'p2'), {'p1': ['v1', 'v2', 'v3', 'v4'], 'p2': ['v1', 'v2', 'v3', 'v4', 'v5']})
tKP.setAllAtOnce([(1,4),
                  (2,1),
                  (6,3),
                  (5,9),
                  (11,3),
                  (1,1),
                  (7,3),
                  (3,6),
                  (5,3),
                  (2,8),
                  (1,9),
                  (5,3),
                  (5,5),
                  (3,1),
                  (7,2),
                  (10,3),
                  (1,4),
                  (8,1),
                  (1,1),
                  (2,7),])

print("Valdo: ", tKP.Valdo())
print("Optimistic: ", tKP.optimistic())
print("Gurwic: ", tKP.Gurwic(0.5))
print("Laplas: ", tKP.Laplas())
print("Bayes: ", tKP.Bayes({'p1': {'p2': {'v1': 0.2, 'v2': 0.2, 'v3': 0.2, 'v4': 0.2, 'v5': 0.2}},
                              'p2': {'p1': {'v1': 0.25, 'v2': 0.25, 'v3': 0.25, 'v4': 0.25}}}))
print("Pareto: ", tKP.Pareto())
print("Nesh: ", tKP.Nesh())
print("Sevij: ", tKP.Sevij())
print("Dominated: ", tKP.dominatedStrat())
print(tKP)

print("This is a Man-Woman-Child case with F-T-A strategies")
tBig = Table(('m', 'w', 'c'), {'m':['f', 't', 'a'],'w':['t','a'],'c':['f','a']})
tBig.setAllAtOnce([(4,3,6), # f t f
                    (2,3,4), # f t a
                    (4,2,6), # f a f
                    (2,5,7), # f a a
                    (3,5,3), # t t f
                    (3,5,4), # t t a
                    (0,2,3), # t a f
                    (0,5,7), # t a a
                    (1,3,3), # a t f
                    (3,3,7), # a t a
                    (4,4,3), # a a f
                    (6,7,10) # a a a
                    ])

print("Valdo: ", tBig.Valdo())
print("Optimistic: ", tBig.optimistic())
print("Gurwic: ", tBig.Gurwic(0.5))
print("Laplas: ", tBig.Laplas())
print("Bayes: ", tBig.Bayes({'m': {'w': {'t': 0.5, 'a': 0.5},
                                   'c': {'f': 0.5, 'a': 0.5}},
                             'w': {'m': {'f': 0.5, 't': 0.5, 'a': 0.5},
                                   'c': {'f': 0.5, 'a': 0.5}},
                             'c': {'m': {'f': 0.5, 't': 0.5, 'a': 0.5},
                                   'w': {'t': 0.5, 'a': 0.5}}}))
print("Sevij: ", tBig.Sevij())
print("Pareto: ", tBig.Pareto())
print("Nesh: ", tBig.Nesh())
tBig.dominatedStrat()
print(tBig)

tBig.setAllAtOnce([(7,3,9), # f f f
               (5,0,1), # f f t
               (5,0,4), # f f a
               (4,3,6), # f t f
               (2,6,4), # f t t
               (2,3,4), # f t a
               (4,2,6), # f a f
               (2,2,1), # f a t
               (2,5,7), # f a a
               (0,1,6), # t f f
               (2,-2,4), # t f t
               (0,-2,4), # t f a
               (3,5,3), # t t f
               (5,8,7), # t t t
               (3,5,4), # t t a
               (0,2,3), # t a f
               (2,2,4), # t a t
               (0,5,7), # t a a
               (1,1,6), # a f f
               (1,-2,1), # a f t
               (3,-2,7), # a f a
               (1,3,3), # a t f
               (1,6,4), # a t t
               (3,3,7), # a t a
               (4,4,3), # a a f
               (4,4,1), # a a t
               (6,7,10) # a a a
               ])

######################

s = linspace(0, 1.5, 21)

res1, res2 = [], []

for x in s:
    for y in s:
        res1.append((1.5 - 0.5 * (x + y) - 1) * x)
        res2.append((1.5 - 0.5 * (x + y) - 1) * y)

data = list(zip(res1,res2))
#print("zip res1res2:", data)
#print("res2:",res2)
print("This is a f1-f2 case with all strategies")
tTest = Table(('f1', 'f2'), {'f1': list(s), 'f2': list(s)})
tTest.setAllAtOnce(data)
print("Valdo: ", tTest.Valdo())
print("Optimistic: ", tTest.optimistic())
print("Gurwic: ", tTest.Gurwic(0.5))
print("Laplas: ", tTest.Laplas())
print("Pareto: ", tTest.Pareto())
print("Nesh: ", tTest.Nesh())
print("Sevij: ", tTest.Sevij())
tTest.dominatedStrat()
print(tTest)


print("This is a M-W case with F-T-A strategies")
tSmall = Table(('m', 'w'), {'m': ['f', 't'], 'w': ['f', 't', 'a']})
tSmall.setAllAtOnce([(8,7),
               (3,3),
               (3,0),
               (0,2),
               (5,8),
               (0,0),])
               #(1,2),
               #(1,3),
               #(6,5)])

print(tSmall)

print("Bayes: ", tSmall.Bayes({'m': {'w': {'f': 0.1, 't': 0.2, 'a': 0.7}},
                               'w': {'m': {'f': 0.2, 't': 0.3}}}))

print("Valdo: ", tSmall.Valdo())
print("Optimistic: ", tSmall.optimistic())
print("Gurwic: ", tSmall.Gurwic(0.5))
print("Laplas: ", tSmall.Laplas())
print("Pareto: ", tSmall.Pareto())
print("Nesh: ", tSmall.Nesh())
print("Sevij: ", tSmall.Sevij())
