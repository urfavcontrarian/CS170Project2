#include <vector>
#include <cmath>
#include <stdexcept>
#include <algorithm>
#include <chrono>
#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <string>
using namespace std;


class NearestNeighborClassifier
{
private:
    vector<vector<double>> trainingData;
    vector<int> trainingLabels;

public:
    void Train(const vector<vector<double>> &instances, const vector<int> &labels)
    {
        trainingData = instances;
        trainingLabels = labels;
    }

    void TrainWithIDs(const vector<int> &instanceIDs, const vector<vector<double>> &fullDataset, const vector<int> &allLabels)
    {
        trainingData.clear(); 
        trainingLabels.clear();
        for (int id : instanceIDs)
        {
            trainingData.push_back(fullDataset[id]);
            trainingLabels.push_back(allLabels[id]);
        }
    }
    
    int Test(const vector<double> &instance) const
    {
        if (trainingData.empty())
        {
            throw runtime_error("Classifier must be trained before testing!");
        }

        double minDistance = 0.0;
        int nearestLabel = -1;

        for (size_t i = 0; i < trainingData.size(); i++)
        {
            double distance = 0.0;
            for (size_t j = 0; j < instance.size(); j++)
            {
                double difference = instance[j] - trainingData[i][j];
                distance += difference * difference;
            }
            distance = sqrt(distance);

            if (distance < minDistance)
            {
                minDistance = distance;
                nearestLabel = trainingLabels[i];
            }
        }

        return nearestLabel;
    }

    int TestWithID(int instanceID, const vector<vector<double>>& fullDataset) const
    {
        return Test(fullDataset[instanceID]);
    }
};


class Validator 
{
    private:
    vector<vector<double>> normalizedData;
    vector<int> labels;
    vector<size_t> featureSubset;
    NearestNeighborClassifier* classifier;
};