import numpy as np
import random as rand
import matplotlib.pyplot as plt
from collections import Counter, defaultdict

DirectionMap = {'N':0, 'S':1, 'E':2, 'W':3}
maxLocations = 41



def make_matrix():

  Breadth =4
  Length = 16

  Collision = defaultdict(lambda: defaultdict(lambda: -1))
  Collision[1][5]=1
  Collision[1][11]=1
  Collision[1][15]=1
  Collision[1][16]=1
  Collision[2][1]=1
  Collision[2][2]=1
  Collision[2][5]=1
  Collision[2][7]=1
  Collision[2][8]=1
  Collision[2][10]=1
  Collision[2][12]=1
  Collision[2][14]=1
  Collision[2][15]=1
  Collision[2][16]=1
  Collision[3][1]=1
  Collision[3][5]=1
  Collision[3][7]=1
  Collision[3][8]=1
  Collision[3][14]=1
  Collision[3][15]=1
  Collision[4][3]=1
  Collision[4][7]=1
  Collision[4][12]=1

  for i in range(1,Breadth+1):
    for j in range(1, Length+1):
      if Collision[i][j]==-1:
        Collision[i][j]=0

  StateFunc=defaultdict(lambda:-1)
  index=1
  for i in range(1,Breadth+1):
    for j in range(1, Length+1):
      if Collision[i][j]==0:
        StateFunc[index]=[i,j]
        index+=1
  return (Collision,StateFunc)





# Building Transition Matrix
def buildTransMat(Collision, StateFunc):

  adjacent_elements = defaultdict(lambda:[])
  for i in range(1,maxLocations +1):
    x,y = StateFunc[i]
    if Collision[x+1][y]==0:
      adjacent_elements[i].append([x+1,y])
    if Collision[x][y+1]==0:
      adjacent_elements[i].append([x,y+1])
    if Collision[x-1][y]==0:
      adjacent_elements[i].append([x-1,y])
    if Collision[x][y-1]==0:
      adjacent_elements[i].append([x,y-1])

  TransMat = np.zeros((maxLocations, maxLocations))
  for i in range(maxLocations):
    for j in range(maxLocations):
      N = len(adjacent_elements[i+1])
      if StateFunc[j+1] in adjacent_elements[i+1]:
        TransMat[i][j]=1/N
  
  return (adjacent_elements, TransMat)




# Building Observation Matrix
def findDiscrepancies(dir, state, adjacent_elements, StateFunc):
  path = bin(dir).replace("0b", "")
  if len(path)==1:
    path= "000"+path
  if len(path)==2:
    path= "00"+path
  if len(path)==3:
    path= "0"+path

  x,y = StateFunc[state]
  Disc_count =0

  for c in ['N','S','E','W']:
    if c == 'N':
      if ([x-1,y] in adjacent_elements[state]) and path[0] == '1':
        Disc_count+=1
      if not([x-1,y] in adjacent_elements[state]) and path[0] == '0':
        Disc_count+=1
    if c == 'E':
      if ([x,y+1] in adjacent_elements[state]) and path[2] == '1':
        Disc_count+=1
      if not([x,y+1] in adjacent_elements[state]) and path[2] == '0':
        Disc_count+=1
    if c == 'W':
      if ([x,y-1] in adjacent_elements[state]) and path[3] == '1':
        Disc_count+=1
      if not([x,y-1] in adjacent_elements[state]) and path[3] == '0':
        Disc_count+=1
    if c == 'S':
      if ([x+1,y] in adjacent_elements[state]) and path[1] == '1':
        Disc_count+=1
      if (not([x+1,y] in adjacent_elements[state])) and path[1] == '0':
        Disc_count+=1
    
  return Disc_count

def buildObservationMatrix(sensorError, adjacent_elements, StateFunc):
  ObsMat = np.zeros((16,maxLocations,maxLocations))
  for dir in range(16):
    sensorList = []
    for state in range(1,maxLocations+1):
      Disc_count = findDiscrepancies(dir, state, adjacent_elements, StateFunc)
      probability=(1-sensorError)**(4-Disc_count)*sensorError**Disc_count
      sensorList.append(probability)
    np.fill_diagonal(ObsMat[dir],sensorList)
  return ObsMat




# To find path
def find_state(x,y, StateFunc):
  for state in range(1,maxLocations+1):
    if StateFunc[state] == [x,y]:
      return state
  return -1



#Find observation Function
def findCurrentObservation(state, Collision, StateFunc ):
  observation = ""
  x,y = StateFunc[state]
  if Collision[x-1][y]==1 or Collision[x-1][y]==-1:
    observation+='N'
  if Collision[x+1][y]==1 or Collision[x+1][y]==-1:
    observation+='S'
  if Collision[x][y+1]==1 or Collision[x][y+1]==-1:
    observation+='E'
  if Collision[x][y-1]==1 or Collision[x][y-1]==-1:
    observation+='W'
  return observation



def constructRandomSequence(Collision, StateFunc):
  evidence = []
  originalPath = []
  NumberRunsForEachLength = 5
  for runLength in range(1,31):
    for numberRun in range(NumberRunsForEachLength):
      startingState = rand.randint(0,1000)%maxLocations
      neighbourList = adjacent_elements[startingState+1]
      pathEvidence = []
      pathOriginal = []
      for step in range(runLength):
        nextdir = rand.randint(0,100)%(len(neighbourList))
        x_nxt, y_nxt = neighbourList[nextdir]
        pathOriginal.append([x_nxt,y_nxt])
        currentState = find_state(x_nxt,y_nxt, StateFunc)
        observation = findCurrentObservation(currentState, Collision, StateFunc)
        pathEvidence.append(observation)
        neighbourList = adjacent_elements[currentState]
      evidence.append(pathEvidence)
      originalPath.append(pathOriginal)
        
  evidence = np.array(evidence,dtype=object)
  originalPath = np.array(originalPath,dtype=object)
  return (evidence, originalPath)




# Filtering the location of a single path
def decodeEvidence(input):
  ans =0
  for c in input:
    if c == 'N':
      ans+=8
    if c == 'E':
      ans+=2
    if c == 'W':
      ans+=1
    if c == 'S':
      ans+=4    
  return ans



def getRandomPathPlots(evidence, TransMat, ObsMat):
  CurrentDistibution = np.full(maxLocations, 1/maxLocations)
  RandomPath = evidence[15]
  plt.plot(CurrentDistibution)
  plt.show()
  for observation in RandomPath:
    observationIndex = decodeEvidence(observation)
    NewDistribution = np.dot(ObsMat[observationIndex],np.dot(TransMat,CurrentDistibution))
    NewDistributionNormalized = NewDistribution/np.sum(NewDistribution)
    CurrentDistibution = NewDistributionNormalized
    plt.plot(CurrentDistibution)
    plt.show()


#Viterbi Algorithm Standard Implementation
def viterbi(TransMat, InitialDistribution, ObsMat, ObservationList):
    NumberStates = TransMat.shape[0]  
    NumberObservations = len(ObservationList)  

    # ConvertObservations
    Observations = []
    for i in range(NumberObservations):
      Observations.append(decodeEvidence(ObservationList[i]))

    # Build ObservationProb
    ObservationProb = np.zeros((NumberStates,16))
    for dir in range(16):
      ObservationProb[:,dir] = ObsMat[dir].diagonal()

    # Initialize D and E matrices
    D = np.zeros((NumberStates, NumberObservations))
    E = np.zeros((NumberStates, NumberObservations-1)).astype(np.int32)
    D[:, 0] = np.multiply(InitialDistribution, ObservationProb[:, 0])

    # Compute D and E in a nested loop
    for n in range(1, NumberObservations):
        for i in range(NumberStates):
            temp_product = np.multiply(TransMat[:, i], D[:, n-1])
            D[i, n] = np.max(temp_product) * ObservationProb[i, Observations[n]]
            E[i, n-1] = np.argmax(temp_product)

    # Backtracking
    S_opt = np.zeros(NumberObservations).astype(np.int32)
    S_opt[-1] = np.argmax(D[:, -1])
    for n in range(NumberObservations-2, -1, -1):
        S_opt[n] = E[int(S_opt[n+1]), n]

    return S_opt, D, E



# Localization error
def LocalizationError(evidence, originalPath, TransMat, ObsMat):
  ErrorValues = []
  for i in range(len(evidence)):
    CurrentDistibution = np.full(maxLocations, 1/maxLocations)
    RandomPath = evidence[i]
    for observation in RandomPath:
      observationIndex = decodeEvidence(observation)
      NewDistribution = np.dot(ObsMat[observationIndex],np.dot(TransMat,CurrentDistibution))
      NewDistributionNormalized = NewDistribution/np.sum(NewDistribution)
      CurrentDistibution = NewDistributionNormalized
    FinalDistribution = list(CurrentDistibution)
    FinalState = FinalDistribution.index(max(FinalDistribution)) 
    x,y = StateFunc[FinalState+1]
    x_orig,y_orig = originalPath[i][-1]
    ManhattanDist = abs(x-x_orig) + abs(y-y_orig)
    ErrorValues.append(ManhattanDist)
  AvgError =  np.zeros(31)
  for i in range(len(evidence)):
    AvgError[len(evidence[i])]+=ErrorValues[i]
  AvgError = AvgError/5
  return AvgError[1:]


# Path Accuracy

def findAccuracy(evidence, originalPath, TransMat, ObsMat):
  AccuracyValues = []
  for i in range(len(evidence)):
    
    InitialDistribution = np.full(maxLocations, 1/maxLocations)
    S_opt, D,E = viterbi(TransMat, InitialDistribution, ObsMat, evidence[i])
    count=0
    for j in range(len(S_opt)):
      if StateFunc[S_opt[j]+1] == originalPath[i][j]:
        count+=1
    accuracy = count/len(S_opt)
    AccuracyValues.append(accuracy)
  AvgAccuracy =  np.zeros(31)
  for i in range(len(evidence)):
    AvgAccuracy[len(evidence[i]) ]+=AccuracyValues[i]
  AvgAccuracy = AvgAccuracy/5
  return AvgAccuracy[1:]


Collision, StateFunc = make_matrix()
adjacent_elements, TransMat = buildTransMat(Collision, StateFunc)
evidence, originalPath = constructRandomSequence(Collision, StateFunc)




ObservationMatrix1 = buildObservationMatrix(0.00, adjacent_elements, StateFunc)
ObservationMatrix2 = buildObservationMatrix(0.02, adjacent_elements, StateFunc)
ObservationMatrix3 = buildObservationMatrix(0.05, adjacent_elements, StateFunc)
ObservationMatrix4 = buildObservationMatrix(0.1, adjacent_elements, StateFunc)
ObservationMatrix5 = buildObservationMatrix(0.2, adjacent_elements, StateFunc)

localizationError = []
localizationError.append(LocalizationError(evidence, originalPath, TransMat, ObservationMatrix1))
localizationError.append(LocalizationError(evidence, originalPath, TransMat, ObservationMatrix2))
localizationError.append(LocalizationError(evidence, originalPath, TransMat, ObservationMatrix3))
localizationError.append(LocalizationError(evidence, originalPath, TransMat, ObservationMatrix4))
localizationError.append(LocalizationError(evidence, originalPath, TransMat, ObservationMatrix5))
localizationError = np.array(localizationError)
plt.plot(localizationError[0])
plt.plot(localizationError[1])
plt.plot(localizationError[2])
plt.plot(localizationError[3])
plt.plot(localizationError[4])
plt.show()

pathaccuracy = []
pathaccuracy.append(findAccuracy(evidence, originalPath, TransMat, ObservationMatrix1))
pathaccuracy.append(findAccuracy(evidence, originalPath, TransMat, ObservationMatrix2))
pathaccuracy.append(findAccuracy(evidence, originalPath, TransMat, ObservationMatrix3))
pathaccuracy.append(findAccuracy(evidence, originalPath, TransMat, ObservationMatrix4))
pathaccuracy.append(findAccuracy(evidence, originalPath, TransMat, ObservationMatrix5))
pathaccuracy = np.array(pathaccuracy)

plt.plot(pathaccuracy[2])
plt.plot(pathaccuracy[3])

plt.show()