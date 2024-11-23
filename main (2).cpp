#include <iostream>
#include <vector>
#include <set>
#include <random>
#include <iomanip>

using namespace std;

double EvaluateFeatureSubset(const set<int> &features)
{
    static random_device randomDevice;
    static mt19937 mersenneTwisterGenerator(randomDevice());
    static uniform_real_distribution<> uniformDistribution(0.0, 100.0);
    return uniformDistribution(mersenneTwisterGenerator);
}

void PrintFeatureSet(const set<int> &features)
{
    cout << "{";
    for (set<int>::iterator iterator = features.begin(); iterator != features.end(); iterator++)
    {
        cout << *iterator;
        if (next(iterator) != features.end())
        {
            cout << ",";
        }
    }
    cout << "}";
}

void ForwardSelection(int totalFeatures)
{
    set<int> currentFeatureSet;
    set<int> bestOverallFeatureSet;
    double bestOverallAccuracy = EvaluateFeatureSubset(currentFeatureSet);
    cout << "Using no features and \"random\" evaluation, I get an accuracy of" << fixed << setprecision(1) << bestOverallAccuracy << "%\n";
    cout << "Beginning search." << endl;

    for (int featureCount = 1; featureCount <= totalFeatures; featureCount++)
    {
        int bestFeatureToAdd = -1;
        double bestAccuracyThisLevel = 0.0;
        set<int> bestFeatureSetThisLevel;

        // Attempt to add features not in the set
        for (int candidateFeature = 1; candidateFeature <= totalFeatures; candidateFeature++)
        {
            if (currentFeatureSet.find(candidateFeature) == currentFeatureSet.end())
            {
                set<int> testFeatureSet = currentFeatureSet;
                testFeatureSet.insert(candidateFeature);

                double currentAccuracy = EvaluateFeatureSubset(testFeatureSet);
                cout << "Using feature(s)";
                PrintFeatureSet(testFeatureSet);
                cout << " accuracy is " << currentAccuracy << "%\n";

                if (currentAccuracy > bestAccuracyThisLevel)
                {
                    bestAccuracyThisLevel = currentAccuracy;
                    bestFeatureToAdd = candidateFeature;
                    bestFeatureSetThisLevel = testFeatureSet;
                }
            }
        }

        if (bestFeatureToAdd != -1)
        {
            currentFeatureSet = bestFeatureSetThisLevel;
            cout << "Feature set";
            PrintFeatureSet(currentFeatureSet);
            cout << " was best, accuracy is" << bestAccuracyThisLevel << "%\n";

            if (bestAccuracyThisLevel > bestOverallAccuracy)
            {
                bestOverallAccuracy = bestAccuracyThisLevel;
                bestOverallFeatureSet = currentFeatureSet;
            }
            else if (featureCount > 1)
            {
                cout << "(Warning, Accuracy has decreased!)" << endl;
            }
        }
    }

    cout << "Finished search!! The best feature subset is ";
    PrintFeatureSet(bestOverallFeatureSet);
    cout << ", which has an accuracy of " << bestOverallAccuracy << "%\n";
}

void BackwardElimination(int totalFeatures)
{
    set<int> currentFeatureSet;

    for (int featureIndex = 1; featureIndex <= totalFeatures; featureIndex++)
    {
        currentFeatureSet.insert(featureIndex);
    }

    set<int> bestOverallFeatureSet = currentFeatureSet;
    double bestOverallAccuracy = EvaluateFeatureSubset(currentFeatureSet);
    cout << "Using no features and \"random\" evaluation, I get an accuracy of" << fixed << setprecision(1) << bestOverallAccuracy << "%\n";
    cout << "Beginning search." << endl;

    for (int featureCount = 1; featureCount <= totalFeatures - 1; featureCount++)
    {
        int bestFeatureToRemove = -1;
        double bestAccuracyThisLevel = 0.0;
        set<int> bestFeatureSetThisLevel;

        // Try to remove each feature that's in the set
        for (set<int>::iterator featureIterator = currentFeatureSet.begin(); featureIterator != currentFeatureSet.end(); featureIterator++)
        {
            set<int> testFeatureSet = currentFeatureSet;
            testFeatureSet.erase(*featureIterator);
            double currentAccuracy = EvaluateFeatureSubset(testFeatureSet);
            cout << "Using feature(s)";
            PrintFeatureSet(testFeatureSet);
            cout << " accuracy is " << currentAccuracy << "%\n";

            if (currentAccuracy > bestAccuracyThisLevel)
            {
                bestAccuracyThisLevel = currentAccuracy;
                bestFeatureToRemove = *featureIterator;
                bestFeatureSetThisLevel = testFeatureSet;
            }
        }

        if (bestFeatureToRemove != -1)
        {
            currentFeatureSet = bestFeatureSetThisLevel;
            cout << "Feature set ";
            PrintFeatureSet(currentFeatureSet);
            cout << " was best, accuracy is " << bestAccuracyThisLevel << "%\n";

            if (bestAccuracyThisLevel > bestOverallAccuracy)
            {
                bestOverallAccuracy = bestAccuracyThisLevel;
                bestOverallFeatureSet = currentFeatureSet;
            }
            else
            {
                cout << "(Warning, Accuracy has decreased!)" << endl;
            }
        }
    }

    cout << "Finished search!! The best feature subset is ";
    PrintFeatureSet(bestOverallFeatureSet);
    cout << ", which has an accuracy of " << bestOverallAccuracy << "%\n";
}

int main()
{
    cout << "Welcome to Feature Selection Algorithm.\n";
    cout << "Please enter total number of features: ";
    int totalFeatures;
    cin >> totalFeatures;

    cout << "Type the number of the algorithm you want to run.\n";
    cout << "1) Forward Selection\n";
    cout << "2) Backward Elimination\n";

    int algorithmChoice;
    cin >> algorithmChoice;

    if (algorithmChoice == 1)
    {
        ForwardSelection(totalFeatures);
    }
    else if (algorithmChoice == 2)
    {
        BackwardElimination(totalFeatures);
    }
    else
    {
        cout << "Invalid choice\n";
    }

    return 0;
}