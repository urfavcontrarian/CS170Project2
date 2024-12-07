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
#include <limits>
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

        double minDistance = numeric_limits<double>::max(); // Initialize to max possible value
        int nearestLabel = trainingLabels[0];

        for (size_t i = 0; i < trainingData.size(); i++)
        {
            double distance = 0.0;
            for (size_t j = 0; j < instance.size(); j++)
            {
                double difference = instance[j] - trainingData[i][j];
                distance += difference * difference;
            }

            if (distance < minDistance)
            {
                minDistance = distance;
                nearestLabel = trainingLabels[i];
            }
        }

        return nearestLabel;
    }

    int TestWithID(int instanceID, const vector<vector<double>> &fullDataset) const
    {
        return Test(fullDataset[instanceID]);
    }
};

class Validator
{
private:
    vector<vector<double>> normalizedData;
    vector<int> labels;
    NearestNeighborClassifier *classifier;

    // Prevent implicit copying
    Validator(const Validator &) = delete;
    Validator &operator=(const Validator &) = delete;

public:
    Validator(const vector<vector<double>> &data, const vector<int> &labels)
    {
        this->normalizedData = normalizeData(data);
        this->labels = labels;
        classifier = new NearestNeighborClassifier();
    }

    vector<vector<double>> normalizeData(const vector<vector<double>> &data)
    {
        auto start = high_resolution_clock::now();
        vector<vector<double>> normalizedData = data;
        size_t numInstances = data.size();
        size_t numFeatures = data[0].size();

        // Normalize each feature (column)
        for (size_t j = 0; j < numFeatures; ++j)
        {
            double minVal = data[0][j];
            double maxVal = data[0][j];

            // Find min and max for this feature
            for (size_t i = 0; i < numInstances; ++i)
            {
                minVal = min(minVal, data[i][j]);
                maxVal = max(maxVal, data[i][j]);
            }

            // Normalize this feature for all instances
            if (maxVal > minVal)
            {
                for (size_t i = 0; i < numInstances; ++i)
                {
                    normalizedData[i][j] = (data[i][j] - minVal) / (maxVal - minVal);
                }
            }
        }
        auto stop = high_resolution_clock::now();
        auto duration = duration_cast<milliseconds>(stop - start);
        cout << "Data normalization duration: " << duration.count() << "ms" << endl;
        return normalizedData;
    }

    double evaluate(const vector<size_t> &featureSubset)
    {
        auto start = high_resolution_clock::now();
        int correctPredictions = 0;
        size_t numInstances = normalizedData.size();

        for (size_t i = 0; i < numInstances; i++)
        {
            vector<double> instance;
            for (size_t j : featureSubset)
            {
                instance.push_back(normalizedData[i][j]);
            }

            vector<vector<double>> trainData;
            vector<int> trainLabels;
            for (int k = 0; k < numInstances; k++)
            {
                if (k != i)
                {
                    vector<double> trainInstance;
                    for (size_t j : featureSubset)
                    {
                        trainInstance.push_back(normalizedData[k][j]);
                    }
                    trainData.push_back(trainInstance);
                    trainLabels.push_back(labels[k]);
                }
            }

            classifier->Train(trainData, trainLabels);
            int predictedLabel = classifier->Test(instance);
            cout << "Instance " << i << "Predicted =" << predictedLabel << "; True = " << labels[i];
            if (predictedLabel == labels[i])
            {
                correctPredictions++;
                cout << " CORRECT" << endl;
                
            }
            else{
                cout << " INCORRECT" << endl;
            }
        }
        auto stop = high_resolution_clock::now();
        auto duration = duration_cast<milliseconds>(stop - start);
        cout << "Data validation duration: " << duration.count() << "ms" << endl;
        return (double)correctPredictions / numInstances;
    }
};

vector<vector<double>> ReadData(const string &filename)
{
    auto start = high_resolution_clock::now();
    vector<vector<double>> data;
    ifstream file(filename);

    if (file.is_open())
    {
        string line;
        while (getline(file, line))
        {
            istringstream iss(line);
            vector<double> instance;
            double value;
            while (iss >> value)
            {
                instance.push_back(value);
            }

            if (!instance.empty())
            {
                data.push_back(instance);
            }
        }
    }

    file.close();
    auto stop = high_resolution_clock::now();
    auto duration = duration_cast<milliseconds>(stop - start);
    cout << "Data parsing duration: " << duration.count() << "ms" << endl;
    return data;
}

int main()
{
    vector<vector<double>> smallData = ReadData("small-test-dataset.txt");
    vector<int> labels;
    vector<vector<double>> largeData = ReadData("large-test-dataset.txt");
    vector<int> labelsL;

    for (const auto &instance : smallData)
    {
        labels.push_back(static_cast<int>(instance[0]));
    }

    for (const auto &instance : largeData)
    {
        labelsL.push_back(static_cast<int>(instance[0]));
    }

    vector<size_t> featureSubset = {3, 5, 7};
    vector<size_t> featureSubsetL = {1, 15, 27};

    Validator validator(smallData, labels);
    Validator validatorL(largeData, labelsL);

    double accuracy = validator.evaluate(featureSubset);
    double accuracyL = validatorL.evaluate(featureSubsetL);

    cout << "Accuracy: " << accuracy << endl;
    cout << "Accuracy for large dataset: " << accuracyL;
}
