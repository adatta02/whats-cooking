import json
import numpy as np
import pandas as pd
import math
import sys
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import MultinomialNB 
from sklearn.feature_selection import VarianceThreshold
from sklearn import svm
from sklearn.decomposition import PCA
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier

RUN_TYPE = "TEST"

def copyAndOutput(predictions, testRecipes):
    for i, w in enumerate(testRecipes):
        testRecipes[i]['cuisine'] = predictions[i]     
    pd.DataFrame(testRecipes, columns=['id', 'cuisine']).to_csv("result.csv", index=False, quoting=3)    
    
def getTestData():    
    if RUN_TYPE == "LIVE":
        data = json.loads('\n'.join(open("test_formatted.json").readlines()))            
        return data    
    else:
        data = json.loads('\n'.join(open("formatted.json").readlines()))        
        num = int(math.floor(len(data) * .75))        
        return data[num:len(data)]                
        
def getRecipeData():
    data = open("formatted.json").readlines()
    data = json.loads('\n'.join(data))
    
    if RUN_TYPE == "LIVE":
        num = len(data)
    else:
        num = int(math.floor(len(data) * .75))
            
    return data[0:num]
        
def outputPercentCorrect(predictions):
    testRecipes = getTestData()
    numCorrect = 0
    
    for i, w in enumerate(testRecipes):
         if 'cuisine' in testRecipes[i] and testRecipes[i]['cuisine'] == predictions[i]:
            numCorrect += 1
    
    percentRight = float(numCorrect) / float(len(testRecipes))
    print "Result: ", percentRight
            
def randomForestDictVector():
    recipeData = getRecipeData()    
    # recipeData = recipeData[0:10000]
    
    labels = [recipe['cuisine'] for recipe in recipeData]
    ingredientsFixtures = [sorted(set(e['ingredients'])) for e in recipeData]
    for i, w in enumerate(ingredientsFixtures):
        ingredientsFixtures[i] = dict(zip(w, [1] * len(w)))        
                
    pipeline = Pipeline([
        ('dict', DictVectorizer()),
        # ('variance', VarianceThreshold()),        
        # ('tfidf', TfidfTransformer()),
        ('bayes', RandomForestClassifier()),
    ])    
    
    pipeline.fit(ingredientsFixtures, labels)
    print pipeline
    
    testRecipes = getTestData()
    testIngredientsFixtures = [sorted(set(e['ingredients'])) for e in testRecipes]
    for i, w in enumerate(testIngredientsFixtures):
        testIngredientsFixtures[i] = dict(zip(w, [1] * len(w)))
        
    predictions = pipeline.predict(testIngredientsFixtures)
    outputPercentCorrect(predictions)
    copyAndOutput(predictions, testRecipes)
    
def svcDictVector():
    recipeData = getRecipeData()
    
    labels = [recipe['cuisine'] for recipe in recipeData]
    ingredientsFixtures = [sorted(set(e['ingredients'])) for e in recipeData]
    for i, w in enumerate(ingredientsFixtures):
        ingredientsFixtures[i] = dict(zip(w, [1] * len(w)))        
                
    pipeline = Pipeline([
        ('dict', DictVectorizer()),
        ('variance', VarianceThreshold()),        
        ('tfidf', TfidfTransformer()),
        ('bayes', svm.LinearSVC()),
    ])    
    
    pipeline.fit(ingredientsFixtures, labels)
    print pipeline
    
    testRecipes = getTestData()    
    testIngredientsFixtures = [sorted(set(e['ingredients'])) for e in testRecipes]
    for i, w in enumerate(testIngredientsFixtures):
        testIngredientsFixtures[i] = dict(zip(w, [1] * len(w)))
        
    predictions = pipeline.predict(testIngredientsFixtures)    
    outputPercentCorrect(predictions)     
    copyAndOutput(predictions, testRecipes)

def buildFeaturesArray(ingredients, recipe):
    res = np.zeros( len(ingredients) )
    for i, w in enumerate(ingredients):
        if w in recipe['ingredients']:
            res[i] = 1
    
    return np.array(res)
    
def bayes():
    recipeData = getRecipeData()
    sel = VarianceThreshold()
        
    ingredients = sorted(set([e for sublist in map(lambda e: e['ingredients'], recipeData) for e in sublist]))
    labels = [recipe['cuisine'] for recipe in recipeData]
    
    features = sel.fit_transform([buildFeaturesArray(ingredients, recipe) for recipe in recipeData])
    ingredients = [ingredients[i] for i in sel.get_support(True)]    
    
    clf = MultinomialNB()
    clf.fit(features, labels)
    
    testRecipes = getTestData()
    testFeatures = [buildFeaturesArray(ingredients, recipe) for recipe in testRecipes]
    predictions = clf.predict(testFeatures)
    
    outputPercentCorrect(predictions)
    copyAndOutput(predictions, testRecipes)

if __name__ == "__main__":
    if len(sys.argv) == 1:
        print "Not enough arguments"
    else:
        getattr(sys.modules[__name__], sys.argv[1])()

