#include "tiff_io.h"

#include <cmath>
#include <vector>
#include <iostream>

void histogram(const string& input_image, const string& histogram_image)
{
    // 1. Read input data
    TiffInfo info = readTiff(input_image);

    // 2. Calculate pixel frequencies
    vector<int> counts(256, 0);
    int max_freq = 0;
    for (unsigned char pixel : info.data) {
        counts[pixel]++;
        if (counts[pixel] > max_freq) {
            max_freq = counts[pixel];
        }
    }

    // 3. Define 512x512 canvas - Initialize with WHITE (255)
    uint32_t h_dim = 512;
    vector<unsigned char> hist_data(h_dim * h_dim, 255);

    // 4. Draw the histogram
    for (int i = 0; i < 256; ++i) {
        // Normalize height to 512 pixels
        int bar_height = 0;
        if (max_freq > 0) {
            bar_height = static_cast<int>((static_cast<double>(counts[i]) / max_freq) * (h_dim - 1));
        }

        // Each of the 256 bins occupies 2 columns
        for (int x_offset = 0; x_offset < 2; ++x_offset) {
            int x = i * 2 + x_offset;

            for (int y = 0; y < bar_height; ++y) {
                // Set bar pixels to BLACK (0)
                hist_data[(h_dim - 1 - y) * h_dim + x] = 0;
            }
        }
    }

    // 5. Setup TiffInfo using your class structure
    TiffInfo outInfo;
    outInfo.width = h_dim;
    outInfo.height = h_dim;
    outInfo.sampleperpixel = 1;
    outInfo.bitperpixel = 8;
    outInfo.photometric = 1;    // MINISBLACK (0=Black, 255=White)
    outInfo.planarconfig = 1;
    outInfo.data = hist_data;

    writeTiff(outInfo, histogram_image);
}