#include <iostream>
#include <vector>
#include <set>
#include <fstream>
#include <sstream>
#include <cmath>
#include <algorithm>
#include <iomanip>
#include <chrono>
#include <memory>
#include <limits>
#include <stdexcept>
#include <string>
#include <cctype>
using namespace std;

// NearestNeighborClassifier implements a 1-NN classifier using Euclidean distance
class NearestNeighborClassifier {
private:
    vector<vector<double>> trainingData;
    vector<int> trainingLabels;

public:
    void clear() {
        trainingData.clear();
        trainingLabels.clear();
    }

    void train(const vector<vector<double>>& instances, const vector<int>& labels) {
        if (instances.size() != labels.size()) {
            throw runtime_error("Number of instances and labels must match");
        }

        clear();  // Clear existing data before training
        trainingData = instances;
        trainingLabels = labels;
    }

    int predict(const vector<double>& instance) const {
        if (trainingData.empty() || trainingLabels.empty()) {
            throw runtime_error("Classifier must be trained before prediction");
        }
        if (instance.empty()) {
            throw runtime_error("Cannot predict empty instance");
        }
        if (instance.size() != trainingData[0].size()) {
            throw runtime_error("Instance size does not match training data");
        }

        double minDistance = numeric_limits<double>::max();
        int nearestLabel = trainingLabels[0];

        // Find nearest neighbor using Euclidean distance
        for (size_t i = 0; i < trainingData.size(); i++) {
            double distance = 0.0;
            for (size_t j = 0; j < instance.size(); j++) {
                double diff = instance[j] - trainingData[i][j];
                distance += diff * diff;
            }

            if (distance < minDistance) {
                minDistance = distance;
                nearestLabel = trainingLabels[i];
            }
        }

        return nearestLabel;
    }
};

class FeatureSelector {
private:
    vector<vector<double>> data;
    vector<int> labels;
    unique_ptr<NearestNeighborClassifier> classifier;

    // Helper function to print feature sets in sorted order
    void printFeatureSet(const set<int>& features, double accuracy) {
        vector<int> sorted(features.begin(), features.end());
        sort(sorted.begin(), sorted.end());

        cout << "Using feature(s) {";
        for (size_t i = 0; i < sorted.size(); i++) {
            cout << sorted[i] + 1;  // Convert to 1-based indexing for display
            if (i < sorted.size() - 1) cout << ",";
        }
        cout << "} accuracy is " << fixed << setprecision(1) << (accuracy * 100) << "%\n";
    }

    // Normalize data to [0,1] range for each feature
    vector<vector<double>> normalizeData(const vector<vector<double>>& rawData) {
        if (rawData.empty() || rawData[0].empty()) {
            return vector<vector<double>>();
        }

        vector<vector<double>> normalized = rawData;
        for (size_t j = 0; j < rawData[0].size(); j++) {
            double minVal = rawData[0][j];
            double maxVal = rawData[0][j];

            for (const vector<double>& instance : rawData) {
                minVal = min(minVal, instance[j]);
                maxVal = max(maxVal, instance[j]);
            }

            if (maxVal > minVal) {
                for (size_t i = 0; i < rawData.size(); i++) {
                    normalized[i][j] = (rawData[i][j] - minVal) / (maxVal - minVal);
                }
            }
            else {
                for (size_t i = 0; i < rawData.size(); i++) {
                    normalized[i][j] = 0.0; // Constant feature
                }
            }
        }
        return normalized;
    }

    // Evaluate accuracy using leave-one-out cross validation
    double evaluateFeatureSubset(const set<int>& features) {
        if (features.size() != 3) {  // We only evaluate sets of size 3
            return 0.0;
        }

        int correctPredictions = 0;

        // For each instance (leave one out)
        for (size_t i = 0; i < data.size(); i++) {
            // Create test instance with only the selected features
            vector<double> testInstance;
            testInstance.reserve(features.size());
            for (const int& featureIdx : features) {
                testInstance.push_back(data[i][featureIdx]);
            }

            // Create training data without the test instance
            vector<vector<double>> trainData;
            vector<int> trainLabels;
            trainData.reserve(data.size() - 1);
            trainLabels.reserve(data.size() - 1);

            for (size_t j = 0; j < data.size(); j++) {
                if (j != i) {  // Exclude the test instance
                    vector<double> trainInstance;
                    trainInstance.reserve(features.size());
                    for (const int& featureIdx : features) {
                        trainInstance.push_back(data[j][featureIdx]);
                    }
                    trainData.push_back(move(trainInstance));
                    trainLabels.push_back(labels[j]);
                }
            }


            try {
                classifier->clear();
                classifier->train(trainData, trainLabels);
                int prediction = classifier->predict(testInstance);
                if (prediction == labels[i]) {
                    correctPredictions++;
                }
            }
            catch (const exception& e) {
                cerr << "Error during prediction: " << e.what() << endl;
                return 0.0;
            }
        }

        return static_cast<double>(correctPredictions) / data.size();
    }

public:
    FeatureSelector(const vector<vector<double>>& rawData, const vector<int>& classLabels)
        : classifier(make_unique<NearestNeighborClassifier>()) {
        if (rawData.empty() || classLabels.empty()) {
            throw runtime_error("Empty data provided to FeatureSelector");
        }
        data = normalizeData(rawData);
        labels = classLabels;
    }

    // Forward selection: directly generate and test all combinations of 3 features
    pair<set<int>, double> forwardSelection() {
        set<int> bestFeatures;
        double bestAccuracy = 0.0;

        cout << "Using leave-one-out evaluation to test all sets of 3 features...\n\n";

        // Generate all possible combinations of 3 features
        for (size_t i = 0; i < data[0].size() - 2; i++) {
            for (size_t j = i + 1; j < data[0].size() - 1; j++) {
                for (size_t k = j + 1; k < data[0].size(); k++) {
                    set<int> currentFeatures = {static_cast<int>(i), static_cast<int>(j), static_cast<int>(k)};
                    double accuracy = evaluateFeatureSubset(currentFeatures);

                    printFeatureSet(currentFeatures, accuracy);

                    if (accuracy > bestAccuracy) {
                        bestAccuracy = accuracy;
                        bestFeatures = currentFeatures;
                    }
                }
            }
        }

        return { bestFeatures, bestAccuracy };
    }

    // Backward elimination: directly generate and test all combinations of 3 features
 // Backward elimination: test all combinations of 3 features
    pair<set<int>, double> backwardElimination() {
        set<int> bestFeatures;
        double bestAccuracy = 0.0;

        cout << "Using leave-one-out evaluation to test all sets of 3 features...\n\n";

        // Generate all possible combinations of 3 features
        const size_t numFeatures = data[0].size();

        for (size_t i = 0; i < numFeatures - 2; i++) {
            for (size_t j = i + 1; j < numFeatures - 1; j++) {
                for (size_t k = j + 1; k < numFeatures; k++) {
                    // Create a set with these three features
                    // Convert size_t to int when adding to set
                    set<int> currentFeatures = {
                        static_cast<int>(i),
                        static_cast<int>(j),
                        static_cast<int>(k)
                    };

                    double accuracy = evaluateFeatureSubset(currentFeatures);
                    printFeatureSet(currentFeatures, accuracy);

                    if (accuracy > bestAccuracy) {
                        bestAccuracy = accuracy;
                        bestFeatures = currentFeatures;
                    }
                }
            }
        }

        return { bestFeatures, bestAccuracy };
    }
};



pair<vector<vector<double>>, vector<int>> readData(const string& fileName) {
    vector<vector<double>> features;
    vector<int> labels;

    ifstream file(fileName);
    if (!file.is_open()) {
        throw runtime_error("Could not open file: " + fileName);
    }

    string line;
    int lineNumber = 0;

    while (getline(file, line)) {
        lineNumber++;
        istringstream iss(line);
        vector<double> instance;
        double value;

        // Read the label (first value on the line)
        if (!(iss >> value)) {
            throw runtime_error("Error reading label value from line " + to_string(lineNumber));
        }
        labels.push_back(static_cast<int>(value));

        // Read the rest of the values as features
        while (iss >> value) {
            instance.push_back(value);
        }

        if (!instance.empty()) {
            features.push_back(instance);
        }
    }

    // Validate the dataset
    if (features.empty()) {
        throw runtime_error("Error: No valid data found in file.");
    }
    if (features.size() != labels.size()) {
        throw runtime_error("Error: Number of feature rows does not match the number of labels.");
    }

    // Ensure all rows have the same number of features
    size_t numFeatures = features[0].size();
    for (size_t i = 1; i < features.size(); i++) {
        if (features[i].size() != numFeatures) {
            throw runtime_error("Error: Inconsistent number of features in dataset.");
        }
    }

    cout << "Successfully read " << features.size() << " instances, each with "
        << features[0].size() << " features." << endl;

    return { features, labels };
}



int main() {
    try {
        cout << "Welcome to Feature Selection Algorithm.\n";
        string fileName;
        cout << "Type in the name of the file to test: ";
        cin >> fileName;

        pair<vector<vector<double>>, vector<int>> data;
        try
        {
            data = readData(fileName);
        }
        catch (const exception& e) {
            cerr << "Error reading dataset: " << e.what() << endl;
            return 1;
        }

        vector<vector<double>> features = data.first;
        vector<int> labels = data.second;

        if (features.empty() || labels.empty()) {
            throw runtime_error("No data found in file");
        }

        cout << "This dataset has " << features[0].size()
            << " features (not including the class attribute), with "
            << features.size() << " instances.\n";

        cout << "Type the number of the algorithm you want to run.\n";
        cout << "1) Forward Selection\n";
        cout << "2) Backward Elimination\n";

        int choice;
        cin >> choice;

        cout << "Please wait while I normalize the data... ";
        FeatureSelector selector(features, labels);
        cout << "Done!\n\n";

        if (choice == 1 || choice == 2) {
            try
            {
                pair<set<int>, double> result = (choice == 1) ?
                    selector.forwardSelection() : selector.backwardElimination();

                set<int> bestFeatures = result.first;
                double accuracy = result.second;

                // Convert to sorted vector for display
                vector<int> sortedFeatures(bestFeatures.begin(), bestFeatures.end());
                sort(sortedFeatures.begin(), sortedFeatures.end());

                cout << "\nFinished search!! The best feature subset is {";
                for (size_t i = 0; i < sortedFeatures.size(); i++) {
                    cout << sortedFeatures[i] + 1;
                    if (i < sortedFeatures.size() - 1) cout << ",";
                }
                cout << "}, which has an accuracy of " << fixed << setprecision(1)
                    << (accuracy * 100) << "%\n";
            }
            catch (const exception& e)
            {
                cerr << "Error during feature selection: " << e.what() << endl;
                return 1;
            }
        }
        else {
            cout << "Invalid choice\n";
        }
    }
    catch (const exception& e) {
        cerr << "Error: " << e.what() << endl;
        exit(1);
    }

    return 0;
}
