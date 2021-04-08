% Testing many machine learning models (built-in from MATLAB)
% with 4 datasets

% Load data
close all;
clear;
clc;
Dataset = input(['Which dataset do you desire to train/test the' ...
            ' models with? \nSelect from 1 to 4, inclusive: ']);  % 1 to 4

if Dataset == 1
    data = importdata('..\Datasets\PhishingData.arff');
%     data = loadARFF('..\Datasets\PhishingData.arff');
    data = data.data;  % converting from struct to matrix
elseif Dataset == 2
    data = importdata('..\Datasets\TrainingDataset.arff');
    data = data.data;  % converting from struct to matrix
elseif Dataset == 3
    data = importdata('..\Datasets\h3cgnj8hft-1\Phishing_Legitimate_full.arff');
    data = data.data;  % converting from struct to matrix
elseif Dataset == 4
    data = readtable('..\Datasets\72ptz43s9v-1\dataset_small.csv');
%     data = readtable('..\Datasets\72ptz43s9v-1\dataset_full.csv');
    data = data{:,:};  % converting from table to matrix
end

