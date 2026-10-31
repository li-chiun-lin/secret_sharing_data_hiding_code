#include "tiff_io.h"

#include <cmath>
#include <iostream>

using namespace std;

WsAnalysisResult compute_ws_analysis(const TiffInfo& info) {
    WsAnalysisResult result;
    if (info.data.empty()) {
        cerr << "Error: Image data is empty." << endl;
        return result;
    }
    if (info.width < 3 || info.height < 3) {
        cerr << "Error: Image is too small for WS analysis." << endl;
        return result;
    }

    int count_flip_better = 0;
    int count_orig_better = 0;
    int samples = 0;
    double total_energy = 0.0;

    for (uint32_t y = 1; y + 1 < info.height; ++y) {
        for (uint32_t x = 1; x + 1 < info.width; ++x) {
            int center = info.data[y * info.width + x];
            int left   = info.data[y * info.width + (x - 1)];
            int right  = info.data[y * info.width + (x + 1)];
            int up     = info.data[(y - 1) * info.width + x];
            int down   = info.data[(y + 1) * info.width + x];

            double predictor = (static_cast<double>(left)
                              + static_cast<double>(right)
                              + static_cast<double>(up)
                              + static_cast<double>(down)) * 0.25;

            double res_orig = static_cast<double>(center) - predictor;
            int flipped = static_cast<int>(center) ^ 1;
            double res_flip = static_cast<double>(flipped) - predictor;

            total_energy += fabs(res_orig);
            if (fabs(res_flip) < fabs(res_orig)) {
                ++count_flip_better;
            } else if (fabs(res_orig) < fabs(res_flip)) {
                ++count_orig_better;
            }
            ++samples;
        }
    }

    result.samples = samples;
    if (samples > 0) {
        result.residual_energy = total_energy / static_cast<double>(samples);
        result.score = static_cast<double>(count_flip_better - count_orig_better)
                       / static_cast<double>(samples);
        result.estimated_rate = static_cast<double>(count_flip_better)
                                / static_cast<double>(samples);
    }
    return result;
}

void ws_analysis(const TiffInfo& info) {
    WsAnalysisResult result = compute_ws_analysis(info);
    cout << "--- WS Analysis for " << info.width << "x" << info.height << " ---" << endl;
    cout << "Samples: " << result.samples << endl;
    cout << "ResidualEnergy: " << result.residual_energy << endl;
    cout << "Score: " << result.score << endl;
    cout << "EstimatedRate: " << result.estimated_rate << endl;
}
