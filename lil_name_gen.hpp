#pragma once

#include <torch/torch.h>

// std
#include <algorithm>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <string>
#include <vector>

class NameGenerator {
   public:
    // Constructor: read file and initialize data
    NameGenerator(const std::string& filename);

    // Generate names using next-token likelihoods
    void generate(int num_names, int seed = 0);

    // Print information about the generated names
    void print_counts_matrix();

    void print_names_info();

   private:
    // Read file and store words in 'words' vector
    void read_file(const std::string& filename);

    // Print file info for debugging purposes
    void print_file_info();

    // Find all unique characters and sort them
    void find_and_sort_characters();

    // Initialize string-to-int and int-to-string maps
    void initialize_maps();

    // Count character pairs
    void count_pairs();

    std::vector<std::string> words;
    std::string all_chars;
    std::map<char, int> stoi;
    std::map<int, char> itos;
    torch::Tensor N;
    torch::Tensor P;
    std::vector<std::string> out;
};