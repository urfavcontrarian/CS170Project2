#include <vector>
#include <cmath>
#include <stdexcept>
#include <algorithm>
#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <string>
#include <limits>
#include <set>
using namespace std;

// The NearestNeighborClassifier implements core classification functionality
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

    void TrainWithIDs(const vector<int> &instanceIDs,
                      const vector<vector<double>> &fullDataset,
                      const vector<int> &allLabels)
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

        double minDistance = numeric_limits<double>::max();
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

// The Validator class handles data preprocessing and evaluation
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

    ~Validator()
    {
        delete classifier;
    }

    vector<vector<double>> normalizeData(const vector<vector<double>> &data)
    {
        vector<vector<double>> normalizedData = data;
        size_t numInstances = data.size();
        size_t numFeatures = data[0].size();

        for (size_t j = 0; j < numFeatures; ++j)
        {
            double minVal = data[0][j];
            double maxVal = data[0][j];

            for (size_t i = 0; i < numInstances; ++i)
            {
                minVal = min(minVal, data[i][j]);
                maxVal = max(maxVal, data[i][j]);
            }

            if (maxVal > minVal)
            {
                for (size_t i = 0; i < numInstances; ++i)
                {
                    normalizedData[i][j] = (data[i][j] - minVal) / (maxVal - minVal);
                }
            }
        }
        return normalizedData;
    }

    double evaluate(const vector<size_t> &featureSubset)
    {
        int correctPredictions = 0;
        size_t numInstances = normalizedData.size();

        // Perform leave-one-out cross validation
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
            if (predictedLabel == labels[i])
            {
                correctPredictions++;
            }
        }

        return static_cast<double>(correctPredictions) / numInstances;
    }

    size_t getNumFeatures() const
    {
        return normalizedData[0].size();
    }
};

// Handles reading different dataset formats
vector<vector<double>> ReadData(const string &fileName)
{
    vector<vector<double>> data;
    ifstream file(fileName);

    if (!file)
    {
        throw runtime_error("Cannot open file: " + fileName);
    }

    // First, check if this is a Titanic dataset by fileName
    bool isTitanic = (fileName == "titanic.txt" || fileName == "titanic-clean.txt");

    if (isTitanic)
    {
        // Handle Titanic format specifically
        vector<double> allValues;
        double value;

        while (file >> value)
        {
            allValues.push_back(value);
        }

        // Process values in groups of 7
        for (size_t i = 0; i < allValues.size(); i += 7)
        {
            if (i + 6 < allValues.size())
            {
                vector<double> instance(allValues.begin() + i, allValues.begin() + i + 7);
                data.push_back(instance);
            }
        }

        cout << "\nTitanic Dataset Features:\n"
             << "1. Passenger Class (1-3)\n"
             << "2. Sex (1 = male, 2 = female)\n"
             << "3. Age\n"
             << "4. Number of Siblings/Spouses\n"
             << "5. Number of Parents/Children\n"
             << "6. Fare\n";
    }
    else
    {
        // Handle standard format (scientific notation)
        string line;
        while (getline(file, line))
        {
            // Skip empty lines
            if (line.empty() || line.find_first_not_of(" \t") == string::npos)
            {
                continue;
            }

            vector<double> instance;
            istringstream iss(line);
            double value;

            // Keep reading values until we can't read anymore
            while (iss >> value)
            {
                instance.push_back(value);
            }

            // Only add non-empty instances
            if (!instance.empty())
            {
                data.push_back(instance);
            }
        }
    }

    if (data.empty())
    {
        throw runtime_error("No valid data found in file");
    }

    // Verify all instances have the same number of values
    size_t numValues = data[0].size();
    for (size_t i = 1; i < data.size(); i++)
    {
        if (data[i].size() != numValues)
        {
            throw runtime_error("Inconsistent number of values across instances");
        }
    }

    cout << "Read " << data.size() << " instances with "
         << data[0].size() << " values each\n";

    return data;
}

int main()
{
    try
    {
        cout << "Welcome to the Feature Selection Program\n\n";
        cout << "Which dataset would you like to analyze?\n";
        cout << "1. Small Dataset (100 instances, 10 features)\n";
        cout << "2. Large Dataset (1000 instances, 40 features)\n";
        cout << "3. Titanic Dataset (survival prediction, 6 features)\n";
        cout << "Enter your choice (1-3): ";
        int choice;
        cin >> choice;
        string fileName;
        int maxFeatures;
        string datasetName;
        int k; // Number of features to select

        // Set parameters based on dataset choice
        switch (choice)
        {
        case 1:
            cout << "Enter filename for small dataset: ";
            cin >> fileName;
            maxFeatures = 10;
            datasetName = "Small";
            k = 3; // Force 3 features for small dataset
            cout << "\nSearching for best subset of 3 features...\n";
            break;
        case 2:
            cout << "Enter filename for large dataset: ";
            cin >> fileName;
            maxFeatures = 40;
            datasetName = "Large";
            k = 3; // Force 3 features for large dataset
            cout << "\nSearching for best subset of 3 features...\n";
            break;
        case 3:
            cout << "Enter fileName for Titanic dataset: ";
            cin >> fileName;
            maxFeatures = 6;
            datasetName = "Titanic";
            cout << "\nHow many features would you like to select (1-" << maxFeatures << "): ";
            cin >> k;
            if (k < 1 || k > maxFeatures)
            {
                throw runtime_error("Invalid number of features specified");
            }
            break;
        default:
            throw runtime_error("Invalid dataset choice");
        }

        cout << "\nSelect search algorithm:\n";
        cout << "1. Forward Selection\n";
        cout << "2. Backward Elimination\n";
        cout << "Enter your choice (1-2): ";
        int algorithmChoice;
        cin >> algorithmChoice;
        // Read and prepare dataset
        vector<vector<double>> data = ReadData(fileName);
        vector<int> labels;

        // Extract labels and remove them from features
        for (const vector<double> &instance : data)
        {
            labels.push_back(static_cast<int>(instance[0]));
        }
        for (vector<double> &instance : data)
        {
            instance.erase(instance.begin());
        }

        // Verify dataset dimensions
        if (choice == 1 && data.size() != 100)
        {
            throw runtime_error("Small dataset must have exactly 100 instances");
        }
        if (choice == 2 && data.size() != 1000)
        {
            throw runtime_error("Large dataset must have exactly 1000 instances");
        }

        // Create validator and initialize feature selection
        Validator validator(data, labels);
        set<int> bestFeatures;
        double bestAccuracy = 0.0;

        if (algorithmChoice == 1)
        {
            // Forward Selection with ordered output
            vector<size_t> currentFeatures; // Use vector instead of set for controlled ordering
            while (currentFeatures.size() < k)
            {
                int bestFeature = -1;
                double bestLocalAcc = 0.0;

                // Try adding each unused feature
                for (size_t i = 0; i < data[0].size(); i++)
                {
                    // Check if feature i is already selected
                    if (find(currentFeatures.begin(), currentFeatures.end(), i) == currentFeatures.end())
                    {
                        vector<size_t> testFeatures = currentFeatures;  // Copy current features
                        testFeatures.push_back(i);                      // Add new feature
                        sort(testFeatures.begin(), testFeatures.end()); // Keep sorted order

                        double acc = validator.evaluate(testFeatures);
                        cout << "Using feature(s) {";
                        for (size_t j = 0; j < testFeatures.size(); j++)
                        {
                            cout << (testFeatures[j] + 1);
                            if (j < testFeatures.size() - 1)
                                cout << ",";
                        }
                        cout << "} accuracy is " << fixed << setprecision(3) << acc << endl;

                        if (acc > bestLocalAcc)
                        {
                            bestLocalAcc = acc;
                            bestFeature = i;
                        }
                    }
                }

                if (bestFeature != -1)
                {
                    currentFeatures.push_back(bestFeature);
                    sort(currentFeatures.begin(), currentFeatures.end()); // Maintain sorted order

                    if (bestLocalAcc > bestAccuracy)
                    {
                        bestAccuracy = bestLocalAcc;
                        bestFeatures.clear();
                        // Convert sorted vector to set
                        for (size_t feature : currentFeatures)
                        {
                            bestFeatures.insert(feature);
                        }
                    }
                    else
                    {
                        cout << "Warning! Accuracy has decreased!\n";
                    }

                    cout << "Feature set {";
                    for (size_t j = 0; j < currentFeatures.size(); j++)
                    {
                        cout << (currentFeatures[j] + 1);
                        if (j < currentFeatures.size() - 1)
                            cout << ",";
                    }
                    cout << "} was best, accuracy is " << fixed << setprecision(3) << bestLocalAcc << endl;
                }
            }
        }
        else if (algorithmChoice == 2)
        {
            // Backward Elimination
            set<int> currentSet;
            for (size_t i = 0; i < data[0].size(); i++)
            {
                currentSet.insert(i);
            }
            bestFeatures = currentSet;
            vector<size_t> allFeatures(currentSet.begin(), currentSet.end());
            bestAccuracy = validator.evaluate(allFeatures);
            while (currentSet.size() > k)
            {
                int featureToRemove = -1;
                double bestLocalAcc = 0.0;

                for (int feature : currentSet)
                {
                    set<int> testSet = currentSet;
                    testSet.erase(feature);
                    vector<size_t> testVector(testSet.begin(), testSet.end());
                    double accuracy = validator.evaluate(testVector);

                    cout << "Using feature(s) {";
                    bool first = true;
                    for (int f : testSet)
                    {
                        if (!first)
                            cout << ",";
                        cout << (f + 1);
                        first = false;
                    }
                    cout << "} accuracy is " << fixed << setprecision(3) << accuracy << endl;

                    if (accuracy > bestLocalAcc)
                    {
                        bestLocalAcc = accuracy;
                        featureToRemove = feature;
                    }
                }

                if (featureToRemove != -1)
                {
                    currentSet.erase(featureToRemove);
                    if (bestLocalAcc > bestAccuracy)
                    {
                        bestAccuracy = bestLocalAcc;
                        bestFeatures = currentSet;
                    }
                    else
                    {
                        cout << "Warning! Accuracy has decreased!\n";
                    }
                }
            }
        }

        // Display final results
        cout << "\nResults for " << datasetName << " Dataset:\n";
        cout << "Best Feature Subset: {";
        int count = 0;
        for (int feature : bestFeatures)
        {
            cout << (feature + 1); // Convert to 1-based indexing for display
            if (count < bestFeatures.size() - 1)
                cout << ", ";
            count++;
        }
        cout << "}\nAccuracy: " << fixed << setprecision(3) << bestAccuracy << "\n";

        // Show reference accuracies for small and large datasets
        if (choice == 1)
        {
            cout << "\nReference: Should find features {3, 5, 7} with accuracy ~0.89\n";
        }
        else if (choice == 2)
        {
            cout << "\nReference: Should find features {1, 15, 27} with accuracy ~0.949\n";
        }
    }
    catch (const exception &e)
    {
        cerr << "Error: " << e.what() << endl;
        exit(1);
    }

    return 0;
}