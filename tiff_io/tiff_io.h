#pragma once

#include <vector>
#include <string>
#include <cstdint> // Required for uint16_t and uint32_t

using namespace std;

typedef unsigned char		Pixel;
typedef vector<Pixel>		Pixels;

class TiffInfo
{
public:
    // Using explicit sizes helps prevent issues with LibTIFF variadic arguments
    uint32_t width;
    uint32_t height;
    uint16_t sampleperpixel;
    uint16_t bitperpixel;
    uint16_t orientation;   // Added field
    uint16_t planarconfig;
    uint16_t photometric;
    int getfield;           // Keep as int for bitmask tracking
    Pixels data;

    TiffInfo() :
        width(0), height(0), sampleperpixel(0), bitperpixel(0),
        orientation(1), // Default to ORIENTATION_TOPLEFT (1)
        planarconfig(0), photometric(0),
        getfield(0)
    {
    }

    void shallowCopy(const TiffInfo& that)
    {
        width = that.width;
        height = that.height;
        sampleperpixel = that.sampleperpixel;
        bitperpixel = that.bitperpixel;
        orientation = that.orientation; // Copy orientation
        planarconfig = that.planarconfig;
        photometric = that.photometric;
        getfield = that.getfield;
        // we don't copy the data here on purpose.
    }

    // Use long long or size_t to prevent overflow for large high-res images
    size_t size() const { return static_cast<size_t>(width) * height; }
};

TiffInfo readTiff(string file);
void writeTiff(const TiffInfo& info, string fn);
void writeTiff(const TiffInfo& info, const Pixels& data, string fn);

// Validation helper moved to tiff_io/validate.cpp
// Validate that a Tiff is 8-bit grayscale and of required size (width x height)
bool validate_gray(const TiffInfo& info, const std::string& name, uint32_t reqWidth, uint32_t reqHeight);

void reverseColor(vector<unsigned char>& data);
void diff(const string& image1, const string& image2, const string& diff_image, const int scale = 1);
void diff_inv(const string& image1, const string& image2, const string& diff_image, const int scale = 1);
void histogram(const string& input_image, const string& histogram_image);
void diff_histogram(const string& image1, const string& image2, const string& histogram_out);

// PSNR helpers (operate on Pixels)
// compute_mse: compute Mean Squared Error (MSE) between two images. Returns -1.0 on error.
// compute_psnr: compute Peak Signal-to-Noise Ratio (PSNR) in decibels. Returns +infinity when MSE == 0,
//               and returns -1.0 on error.
double compute_mse(const Pixels &a, const Pixels &b);
double compute_psnr(const Pixels &a, const Pixels &b);
double compute_ssim(const Pixels& a, const Pixels& b);

struct RsAnalysisResult {
    int total_blocks = 0;
    int regular_m = 0;
    int singular_m = 0;
    int regular_neg_m = 0;
    int singular_neg_m = 0;
    double regular_m_ratio = 0.0;
    double singular_m_ratio = 0.0;
    double regular_neg_m_ratio = 0.0;
    double singular_neg_m_ratio = 0.0;
    double gap = 0.0;
};

struct WsAnalysisResult {
    double estimated_rate = 0.0;
    double residual_energy = 0.0;
    double score = 0.0;
    int samples = 0;
};

struct SpaAnalysisResult {
    double estimated_rate = 0.0;
    double pair_imbalance = 0.0;
    double score = 0.0;
    int pairs = 0;
};

RsAnalysisResult compute_rs_analysis(const TiffInfo& info);
WsAnalysisResult compute_ws_analysis(const TiffInfo& info);
SpaAnalysisResult compute_spa_analysis(const TiffInfo& info);
void rs_analysis(const TiffInfo& info);
void ws_analysis(const TiffInfo& info);
void spa_analysis(const TiffInfo& info);
