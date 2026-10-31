#include "tiff_io.h"

#include <cmath>
#include <vector>
#include <iostream>

void reverseColor(vector<unsigned char>& data)
{
    for (auto& pixel : data) {
        pixel = 255 - pixel;
    }
}

// Computes absolute difference * 20, clamped to 255
vector<unsigned char> _diff_worker(const vector<unsigned char>& data1,
    const vector<unsigned char>& data2, int scale = 1)
{
    if (data1.size() != data2.size())
        throw runtime_error("Data mismatch: different sizes");

    size_t length = data1.size();
    vector<unsigned char> diff_data(length);

    for (size_t i = 0; i < length; ++i) {
        int d = static_cast<int>(data1[i]) - static_cast<int>(data2[i]);
        diff_data[i] = static_cast<unsigned char>(std::min(255, abs(d) * scale));
    }

    return diff_data;
}

void diff(const string& image1, const string& image2, const string& diff_image, const int scale)
{
    TiffInfo info1 = readTiff(image1);
    TiffInfo info2 = readTiff(image2);

    if (info1.width != info2.width || info1.height != info2.height)
        throw runtime_error("Dimension mismatch: Images must be same size");

    // 1. Compute difference
    vector<unsigned char> diff_data = _diff_worker(info1.data, info2.data, scale);

    // 2. Write the diff image as is
    TiffInfo outInfo;
    outInfo.shallowCopy(info1);
    outInfo.data = std::move(diff_data);

    writeTiff(outInfo, diff_image);
}

void diff_inv(const string& image1, const string& image2, const string& diff_image, const int scale)
{
    TiffInfo info1 = readTiff(image1);
    TiffInfo info2 = readTiff(image2);

    if (info1.width != info2.width || info1.height != info2.height)
        throw runtime_error("Dimension mismatch: Images must be same size");

    // 1. Compute difference
    vector<unsigned char> diff_data = _diff_worker(info1.data, info2.data, scale);

    // 2. Reverse color (high difference -> dark)
    reverseColor(diff_data);

    // 3. Write the diff image
    TiffInfo outInfo;
    outInfo.shallowCopy(info1);
    outInfo.data = std::move(diff_data);

    writeTiff(outInfo, diff_image);
}

/**
 * Generates a histogram representing the difference between two images.
 * 1. Computes the absolute difference (scaled by 20).
 * 2. Calculates frequencies of those difference values.
 * 3. Draws and saves a 512x512 histogram TIFF.
 */
void diff_histogram(const string& image1, const string& image2, const string& histogram_out)
{
    // 1. Read both input images
    TiffInfo info1 = readTiff(image1);
    TiffInfo info2 = readTiff(image2);

    if (info1.width != info2.width || info1.height != info2.height) {
        throw runtime_error("Dimension mismatch: Images must be the same size to compute difference.");
    }

    // 2. Compute the difference data using your existing worker
    // This applies the |d| * 20 logic
    vector<unsigned char> diff_data = _diff_worker(info1.data, info2.data);

    // 3. Calculate pixel frequencies for the difference data
    vector<int> counts(256, 0);
    int max_freq = 0;
    for (unsigned char pixel : diff_data) {
        counts[pixel]++;
        if (counts[pixel] > max_freq) {
            max_freq = counts[pixel];
        }
    }

    // 4. Define 512x512 canvas - Initialize with WHITE (255)
    uint32_t h_dim = 512;
    vector<unsigned char> hist_data(h_dim * h_dim, 255);

    // 5. Draw the histogram bars
    for (int i = 0; i < 256; ++i) {
        int bar_height = 0;
        if (max_freq > 0) {
            // Normalize height relative to the max frequency
            bar_height = static_cast<int>((static_cast<double>(counts[i]) / max_freq) * (h_dim - 1));
        }

        // Each bin is 2 pixels wide to fill the 512 width
        for (int x_offset = 0; x_offset < 2; ++x_offset) {
            int x = (i * 2) + x_offset;
            for (int y = 0; y < bar_height; ++y) {
                // Fill from bottom up: (h_dim - 1 - y)
                hist_data[(h_dim - 1 - y) * h_dim + x] = 0; // Black bars
            }
        }
    }

    // 6. Setup output metadata and write to file
    TiffInfo outInfo;
    outInfo.width = h_dim;
    outInfo.height = h_dim;
    outInfo.sampleperpixel = 1;
    outInfo.bitperpixel = 8;
    outInfo.photometric = 1; // MINISBLACK
    outInfo.planarconfig = 1;
    outInfo.data = std::move(hist_data);

    writeTiff(outInfo, histogram_out);
}
