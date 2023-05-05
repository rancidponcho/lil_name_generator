#include "lil_name_gen.hpp"

NameGenerator::NameGenerator(const std::string& filename) : N{torch::zeros({27, 27}, torch::kInt32)} {
    read_file(filename);
    print_file_info();
    find_and_sort_characters();
    initialize_maps();
    count_pairs();
}

// use next-token liklihoods to generate names
void NameGenerator::generate(int num_names, int seed) {
    int ix = 0;
    auto g = torch::make_generator<torch::CPUGeneratorImpl>(seed);
    P = (N + 1).to(torch::kFloat);  // add-one smoothing
    P = P / P.sum(1, true);         // normalize rows to find probabilities
    for (int i = 0; i < num_names; i++) {
        std::string current_name = "";
        while (true) {                                                 // iterate until name ends
            auto p = P[ix];                                            // get starting letters
            ix = torch::multinomial(p, 1, true, g)[0].item().toInt();  // draw sample from distribution ([0] to get scalar)
            if (ix == 0) {                                             // end of word
                out.push_back(current_name);                           // add name to vector
                break;
            } else {
                current_name.push_back(itos[ix]);  // otherwise, add char to name
            }
        }
    }
    std::cout << "GENERATED NAMES ----------------------\n"
              << out << std::endl;

    // CALCULATE LIKELIHOODS
    int n = 0;
    float log_likelihood = 0;
    for (const auto& name : out) {
        std::vector<char> chs = {'.'};
        chs.insert(chs.end(), name.begin(), name.end());
        chs.push_back('.');
        for (auto it = chs.begin(); it != chs.end() - 1; it++) {
            auto log_prob = torch::log(P[stoi[*it]][stoi[*(it + 1)]]);
            log_likelihood += log_prob.item().toFloat();
            n += 1;
        }
    }
    auto nll = -log_likelihood;
    std::cout << "NEGATIVE LOG LIKELIHOOD (NLL): " << nll << std::endl;
    std::cout << "NORMALIZED NLL: " << nll / n << std::endl;
}

void NameGenerator::read_file(const std::string& filename) {
    // READ FILE
    std::ifstream file(filename);
    if (file.is_open()) {
        std::string line;
        while (std::getline(file, line)) {
            words.push_back(line);
        }
        file.close();
    } else {
        std::cout << " unable to open file";
    }
}

void NameGenerator::find_and_sort_characters() {
    for (const auto& word : words) {
        all_chars.append(word);
    }
    std::sort(all_chars.begin(), all_chars.end());
    auto last = std::unique(all_chars.begin(), all_chars.end());
    all_chars.erase(last, all_chars.end());
}

void NameGenerator::initialize_maps() {
    // STRINT TO INT (STOI)
    int index = 1;
    for (const auto& ch : all_chars) {
        stoi[ch] = index++;
    }
    stoi['.'] = 0;

    // INT TO STRING (ITOS)
    for (const auto& p : stoi) {
        itos[p.second] = p.first;
    }
}

void NameGenerator::count_pairs() {
    for (const auto& w : words) {
        std::vector<char> chs = {'.'};
        chs.insert(chs.end(), w.begin(), w.end());
        chs.push_back('.');

        for (auto it = chs.begin(); it != chs.end() - 1; it++) {
            N[stoi[*it]][stoi[*(it + 1)]] += 1;
        }
    }
}

void NameGenerator::print_file_info() {
    std::cout << "FILEINFO -----------------------------" << std::endl;
    std::cout << "CONTENTS: ";
    for (int i = 0; i < 10; i++) {  // print first 10 names
        std::cout << words[i];
        if (i < 9) {
            std::cout << ", ";
        }
    }
    std::cout << "..." << std::endl;
    std::cout << "NUMBER OF NAMES: " << words.size() << std::endl;
    auto longest_length = std::max_element(words.begin(), words.end(), [](const std::string& a, const std::string& b) { return a.length() < b.length(); });
    std::cout << "LONGEST: " << longest_length->length() << std::endl;
    auto shortest_length = std::min_element(words.begin(), words.end(), [](const std::string& a, const std::string& b) { return a.length() < b.length(); });
    std::cout << "SHORTEST: " << shortest_length->length() << std::endl;
    std::cout << "CHARS: " << all_chars << std::endl;
}

void NameGenerator::print_counts_matrix() {
    std::cout << "COUNTS MATRIX ------------------------ " << std::endl;
    for (int i = 0; i < N.size(0); i++) {
        for (int j = 0; j < N.size(1); j++) {
            std::cout << std::setw(4) << itos[i] << itos[j];  // Use std::setw to set a fixed width
        }
        std::cout << std::endl;
        for (int j = 0; j < N.size(1); j++) {
            std::cout << std::setw(5) << N[i][j].item();  // Use std::setw to set a fixed width
        }
        std::cout << std::endl;
    }
}

void NameGenerator::print_names_info() {
    std::cout << "NAMES INFO ---------------------------" << std::endl;
    // CALCULATE LIKELIHOODS IN GENERATED NAMES
    int n = 0;
    for (const auto& name : out) {
        std::vector<char> chs = {'.'};
        chs.insert(chs.end(), name.begin(), name.end());
        chs.push_back('.');
        for (auto it = chs.begin(); it != chs.end() - 1; it++) {
            auto ix = stoi[*it];
            auto iy = stoi[*(it + 1)];
            auto prob = P[ix][iy];
            auto log_prob = torch::log(prob);
            std::cout << *it << *(it + 1) << " : " << std::setw(10) << prob.item() << ' ' << log_prob.item() << std::endl;
            n += 1;
        }
    }
}
