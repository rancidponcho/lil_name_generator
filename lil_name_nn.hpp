/* lil_name_nn.hpp
 * Class for training a single-layer feed-forward neural network to generate names.
 * TODO: Tranfer tensors to device. Implement generator instead of manual seed. Current
 * implementation causes irregularities in the generated names. Create a print info function.
 */

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

class LilNameNN {
   public:
    LilNameNN(int seed = 0) {
        // read file and store words in vector
        std::vector<std::string> words;
        readFile("../names.txt", words);

        // initialize maps for encoding and decoding characters
        initMaps(words, ctoi, itoc);

        // CREATE TRAINING SET BIGRAMS
        num_pairs = 0;
        for (const auto& w : words) {
            num_pairs += w.size() + 1;
        }

        xs = torch::empty({num_pairs}, torch::kInt64);
        ys = torch::empty({num_pairs}, torch::kInt64);

        createTrainingSet(words);

        torch::manual_seed(seed);
    };

    ~LilNameNN(){};

    void train(int iters) {
        // INIT SINGLE-LAYER FEED-FORWARD
        int64_t unique_chars = ctoi.size();
        auto options = torch::TensorOptions().requires_grad(true);     // let torch know we want to compute gradients
        auto W = torch::randn({unique_chars, unique_chars}, options);  // random weights with normal distribution for unique_chars # of neurons
        // GRADIENT DESCENT
        for (int i = 0; i < iters; i++) {
            // forward pass
            auto xenc = torch::one_hot(xs, unique_chars).to(torch::kFloat);  // 1) encode one-hot format
            auto logits = torch::matmul(xenc, W);                            // 2) predict log-counts: dot product for each element
            auto counts = logits.exp();                                      // 3) SOFTMAX: exponentiating allows for negative values and
            auto probs = counts / counts.sum(1, true);                       // hyperbolates likelihoods. Row normalization to get probabilities
            // loss
            auto indices = torch::arange(num_pairs);  // for indexing through prob of each bigram
            /*
             * Compute the negative log likelihood loss for each letter in the word.
             * 1. Select probabilities corresponding to the indices from the 'probs' tensor.
             * 2. Gather the probabilities from 'ys' tensor by reshaping it to a column vector.
             * 3. Squeeze the tensor to remove dimensions of size 1.
             * 4. Calculate the natural logarithm of each probability.
             * 5. Compute the mean of the resulting tensor.
             * 6. Add L2 regularization to prevent overfitting.
             */
            auto loss = -probs.index_select(0, indices)
                             .gather(1, ys.view({-1, 1}))
                             .squeeze()
                             .log()
                             .mean() +
                        W.pow(2).mean() * 1e-3;  // L2 regularization
            std::cout << "[" << i << "]: " << loss.item() << std::endl;

            // backward pass
            loss.backward();  // compute gradients

            // update weights
            W.data() += -50 * W.grad();  // learning rate * gradient

            // reset gradients to avoid accumulation
            W.grad().zero_();
        }

        // SAVE MODEL
        torch::save(W, "../model.pt");
    }

    void sample(int num_names) {
        // LOAD MODEL
        torch::Tensor W;
        torch::load(W, "../model.pt");
        // SAMPLE MODEL
        std::vector<std::string> out;
        int ix = 0;
        int64_t unique_chars = ctoi.size();
        for (int i = 0; i < 10; i++) {
            std::string current_name = "";
            while (true) {  // iterate until name ends
                auto xenc = torch::one_hot(torch::tensor({ix}), unique_chars).to(torch::kFloat);
                auto logits = torch::matmul(xenc, W);
                auto counts = logits.exp();
                auto p = counts / counts.sum(1, true);
                ix = torch::multinomial(p, 1, true)[0].item().toInt();  // draw sample from distribution ([0] to get scalar)
                if (ix == 0) {                                          // end of word
                    out.push_back(current_name);                        // add name to vector
                    break;
                } else {
                    current_name.push_back(itoc[ix]);  // otherwise, add char to name
                }
            }
        }

        std::cout << "GENERATED NAMES ----------------------\n";
        for (const auto& name : out) {
            std::cout << name << std::endl;
        }

        // BIGRAM PRINT INFO
        // auto nlls = torch::zeros({5});  // negative log-likelihoods
        // for (int i = 0; i < 5; i++) {
        //     auto x = xs[i].item().toInt();  // input index  // can probably convert toInt before for loop
        //     auto y = ys[i].item().toInt();  // predicted index
        //     std::cout << "--------" << std::endl;
        //     std::cout << "[" << i << "] \'" << itoc[x] << itoc[y] << "\' " << "[" << x << "]" << "[" << y << "]" << std::endl;
        //     std::cout << "Probabilities: " << std::endl;
        //     for(int j = 0; j < probs.size(1); j++) {
        //         std::cout << probs[i][j].item().toFloat() << " ";
        //     }
        //     std::cout << std::endl;
        //     std::cout << "Probability of next character: " << probs[i][y].item().toFloat() * 100.0f << "%" << std::endl;
        //     std::cout << "log-likelihood: " << probs[i][y].log().item() << std::endl;
        //     std::cout << "NLL: " << -probs[i][y].log().item() << std::endl;
        //     nlls[i] = -probs[i][y].log().item();
        // }
        // std::cout << "========" << std::endl;
        // std::cout << "LOSS (average nll): " << nlls.mean().item() << std::endl;
    }

   private:
    int64_t num_pairs;
    std::map<char, int> ctoi;
    std::map<int, char> itoc;
    torch::Tensor xs;
    torch::Tensor ys;

    // Read file and store words in 'words' vector
    void readFile(const std::string& filename, std::vector<std::string>& words) {
        std::ifstream file(filename);
        if (file.is_open()) {
            std::string line;
            while (std::getline(file, line)) {
                words.push_back(line);
            }
            file.close();
        } else {
            std::cout << "unable to open file: " << filename << std::endl;
        }
    }

    // Initializes character to int and int to character maps
    // used to encode and decode for one-hot
    void initMaps(const std::vector<std::string>& words, std::map<char, int>& ctoi, std::map<int, char>& itoc) {
        // Sort characters and remove duplicates
        // std::unique requires sorted input
        std::string all_chars;
        for (const auto& word : words) {
            all_chars.append(word);
        }
        std::sort(all_chars.begin(), all_chars.end());
        auto last = std::unique(all_chars.begin(), all_chars.end());
        all_chars.erase(last, all_chars.end());

        // INITIALIZE MAPS
        // char to int
        ctoi['.'] = 0;
        int index = 1;
        for (const auto& ch : all_chars) {
            ctoi[ch] = index++;
        }
        // char to string
        for (const auto& p : ctoi) {
            itoc[p.second] = p.first;
        }
    }

    void createTrainingSet(const std::vector<std::string>& words) {
        int64_t idx = 0;
        for (const auto& w : words) {                       // for each word
            std::vector<char> chars = {'.'};                // start with '.'
            chars.insert(chars.end(), w.begin(), w.end());  // insert word
            chars.push_back('.');                           // end with '.'

            for (auto it = chars.begin(); it < chars.end() - 1; it++) {  // for each character
                int ix1 = ctoi[*it];                                     // convert to int
                int ix2 = ctoi[*(it + 1)];                               //
                xs.index_put_({idx}, ix1);                               // Set values directly using the index
                ys.index_put_({idx}, ix2);

                idx++;
            }
        }
    }
};