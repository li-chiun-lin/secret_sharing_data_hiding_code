#include "tiff_io.h"

#include <iostream>

bool validate_gray(const TiffInfo& info, const std::string& name, uint32_t reqWidth, uint32_t reqHeight)
{
    if (info.data.empty()) {
        std::cerr << "Error: image '" << name << "' has no data\n";
        return false;
    }
    if (info.sampleperpixel != 1) {
        std::cerr << "Error: image '" << name << "' is not grayscale (sampleperpixel=" << info.sampleperpixel << ")\n";
        return false;
    }
    if (info.bitperpixel != 8) {
        std::cerr << "Error: image '" << name << "' is not 8-bit (bitperpixel=" << info.bitperpixel << ")\n";
        return false;
    }
    if (info.width != reqWidth || info.height != reqHeight) {
        std::cerr << "Error: image '" << name << "' is not " << reqWidth << "x" << reqHeight
                  << " (width=" << info.width << ", height=" << info.height << ")\n";
        return false;
    }
    size_t expectedSize = static_cast<size_t>(info.width) * static_cast<size_t>(info.height);
    if (info.data.size() != expectedSize) {
        std::cerr << "Error: image '" << name << "' data size mismatch (data.size()=" << info.data.size()
                  << ", expected=" << expectedSize << ")\n";
        return false;
    }
    return true;
}