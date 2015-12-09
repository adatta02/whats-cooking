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

def svc():
    data = open("formatted.json").readlines()
    data = '\n'.join(data)    
    recipeData = json.loads(data)        
    
    recipeData = recipeData[0:5000]
    ingredients = sorted(set([e for sublist in map(lambda e: e['ingredients'], recipeData) for e in sublist]))
    labels = [recipe['cuisine'] for recipe in recipeData]
    features = [buildFeaturesArray(ingredients, recipe) for recipe in recipeData]
    
    print "Loaded " + str(len(recipeData)) + " with " + str(len(ingredients)) + " ingredients"    
        
    pca = PCA(100)
    pca.fit(features)
        
    print "PCA: " + pca
        
    clf = svm.SVC()
    clf.fit(pca.fit_transform(features), labels)
    
    print "CLF: " + clf
    
    testData = '\n'.join(open("test_formatted.json").readlines())
    testRecipes = json.loads(testData)    
    testFeatures = [buildFeaturesArray(ingredients, recipe) for recipe in testRecipes]
    
    predictions = clf.predict( pca.fit_transform(testFeatures) )
    
    for i, w in enumerate(testRecipes):
        testRecipes[i]['cuisine'] = predictions[i]
    
    pd.DataFrame(testRecipes, columns=['id', 'cuisine']).to_csv("result.csv", index=False, quoting=3)
            

def buildFeaturesArray(ingredients, recipe):
    res = np.zeros( len(ingredients) )
    for i, w in enumerate(ingredients):
        if w in recipe['ingredients']:
            res[i] = 1
    
    return np.array(res)
    
def bernouli():
    data = open("formatted.json").readlines()
    data = '\n'.join(data)
    recipeData = json.loads(data)
    sel = VarianceThreshold()
        
    ingredients = sorted(set([e for sublist in map(lambda e: e['ingredients'], recipeData) for e in sublist]))
    labels = [recipe['cuisine'] for recipe in recipeData]
    
    features = sel.fit_transform([buildFeaturesArray(ingredients, recipe) for recipe in recipeData])
    ingredients = [ingredients[i] for i in sel.get_support(True)]    
    
    clf = MultinomialNB()
    clf.fit(features, labels)
    
    testData = '\n'.join(open("test_formatted.json").readlines())
    testRecipes = json.loads(testData)
    # testRecipes = testRecipes[0:1000]
    
    testFeatures = [buildFeaturesArray(ingredients, recipe) for recipe in testRecipes]
    predictions = clf.predict(testFeatures)
    
    for i, w in enumerate(testRecipes):
        testRecipes[i]['cuisine'] = predictions[i]
    
    pd.DataFrame(testRecipes, columns=['id', 'cuisine']).to_csv("result.csv", index=False, quoting=3)

if __name__ == "__main__":
    if len(sys.argv) == 1:
        print "Not enough arguments"
    else:
        getattr(sys.modules[__name__], sys.argv[1])()

