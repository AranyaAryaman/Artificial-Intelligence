import matplotlib.pyplot as plt
import matplotlib.image
import itertools as it
import pomegranate as pom
import pygraphviz
import tempfile


F = 'Fraud'
T = 'Travel'
OD = 'OwnsDevice'
FP = 'ForeignPurchase'
OP = 'OnlinePurchase'

travelDist = pom.DiscreteDistribution({False: 0.05, True: 0.95})

foreignPurchaseDist = pom.ConditionalProbabilityTable(
    [
		[False, False, 0.9999],
	 	[False, True, 0.0001],
		[True, False, 0.12],
     	[True, True, 0.88]
	],
	[travelDist])

ownsDeviceDist = pom.DiscreteDistribution({False: 0.3, True: 0.7})

onlinePurchaseDist = pom.ConditionalProbabilityTable(
    [
		[False, False, 0.9995],
		[False, True, 0.0005],
		[True, False, 0.60],
     	[True, True, 0.40]
	],
	[ownsDeviceDist])

fraudDist = pom.ConditionalProbabilityTable(
    [
		[False, False, False, 0.25],
		[False, True, False, 0.15],
     	[True, False, False, 0.20],
		[True, True, False, 0.0005],
     	[False, False, True, 0.75],
		[False, True, True, 0.85],
     	[True, False, True, 0.80],
		[True, True, True, 0.9995]
	],
    [onlinePurchaseDist, travelDist])

foreignPurchase = pom.Node(foreignPurchaseDist, name=FP)
onlinePurchase = pom.Node(onlinePurchaseDist, name=OP)
fraud = pom.Node(fraudDist, name=F)
ownsDevice = pom.Node(ownsDeviceDist, name=OD)
travel = pom.Node(travelDist, name=T)


model = pom.BayesianNetwork("Fraud Detection")
model.add_states(fraud, ownsDevice, travel, foreignPurchase, onlinePurchase)
model.add_edge(travel, foreignPurchase)
model.add_edge(ownsDevice, onlinePurchase)
model.add_edge(travel, fraud)
model.add_edge(onlinePurchase, fraud)
model.bake()


def plot(model, filename=None):
    G = pygraphviz.AGraph(directed=True)

    for state in model.states:
        G.add_node(state.name, color='red')

    for parent, child in model.edges:
        G.add_edge(parent.name, child.name)

    if filename is None:
        with tempfile.NamedTemporaryFile() as tf:
            G.draw(tf, format='png', prog='dot')
            img = matplotlib.image.imread(tf)
            plt.imshow(img)
            plt.axis('off')
            plt.show()
    else:
        G.draw(filename, format='pdf', prog='dot')


##### Part 1 Model plot and (conditional) probability tables #####

model.plot()
plt.show()

##### Part 2 Gibbs Sampling #####

N = 10000

predictions = model._gibbs(n=N, evidences=[{}])
print(len(list(filter(lambda x: x[0], predictions))) / N)

predictions = model._gibbs(n=N, evidences=[{OD: True}])
print(len(list(filter(lambda x: x[0], predictions))) / N)

predictions = model._gibbs(n=N, evidences=[{OD: True, T: True}])
print(len(list(filter(lambda x: x[0], predictions))) / N)