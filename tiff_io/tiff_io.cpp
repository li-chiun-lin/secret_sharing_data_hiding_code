#include "tiff_io.h"

#include <iostream>
#include "tiffio.h"
#include "tiff.h"

TiffInfo readTiff(string file)
{
	TiffInfo info;

	TIFF* tif = TIFFOpen(file.c_str(), "r");
	if (tif == NULL)
		return info;

	// Temporary variables matching LibTIFF types
	uint32_t w32, h32;
	uint16_t s16, b16, p16, c16, o16;

	// Reset ret
	int ret = 0;

	// Mandatory Tags
	if (TIFFGetField(tif, TIFFTAG_IMAGEWIDTH, &w32)) {
		info.width = w32;
		ret |= (1 << 1);
	}
	if (TIFFGetField(tif, TIFFTAG_IMAGELENGTH, &h32)) {
		info.height = h32;
		ret |= (1 << 2);
	}

	// Optional Tags with standard defaults
	if (TIFFGetField(tif, TIFFTAG_SAMPLESPERPIXEL, &s16)) {
		info.sampleperpixel = s16;
		ret |= (1 << 0);
	}
	else {
		info.sampleperpixel = 1; // Default for grayscale
	}

	if (TIFFGetField(tif, TIFFTAG_BITSPERSAMPLE, &b16)) {
		info.bitperpixel = b16;
		ret |= (1 << 3);
	}

	// Fixed Orientation logic (avoids the crash/error)
	if (TIFFGetField(tif, TIFFTAG_ORIENTATION, &o16)) {
		info.orientation = o16;
		ret |= (1 << 4);
	}
	else {
		info.orientation = ORIENTATION_TOPLEFT; // Most common default
	}

	if (TIFFGetField(tif, TIFFTAG_PLANARCONFIG, &c16)) {
		info.planarconfig = c16;
		ret |= (1 << 5);
	}

	if (TIFFGetField(tif, TIFFTAG_PHOTOMETRIC, &p16)) {
		info.photometric = p16;
		ret |= (1 << 6);
	}

	info.getfield = ret;

	tmsize_t sls = TIFFScanlineSize(tif);
	tdata_t buf_tif = _TIFFmalloc(sls);

	for (uint32_t row = 0; row < info.height; row++)
	{
		TIFFReadScanline(tif, buf_tif, row);
		unsigned char* data = (unsigned char*)buf_tif;

		info.data.insert(info.data.end(), data, data + info.width);
	}

	_TIFFfree(buf_tif);
	TIFFClose(tif);

	return info;
}

void writeTiff(const TiffInfo& info, const Pixels& data, string fn)
{
	if (fn.empty())
		return;

	TiffInfo img;
	img.shallowCopy(info);
	img.data = data;
	writeTiff(img, fn);
}

void writeTiff(const TiffInfo& info, string fn)
{
	if (fn.empty())
		return;

	TIFF* out = TIFFOpen(fn.c_str(), "w");
	if (!out) {
		std::cerr << "writeTiff: TIFFOpen failed for " << fn << "\n";
		return;
	}

	TIFFSetField(out, TIFFTAG_IMAGEWIDTH, info.width);
	TIFFSetField(out, TIFFTAG_IMAGELENGTH, info.height);
	TIFFSetField(out, TIFFTAG_SAMPLESPERPIXEL, info.sampleperpixel);
	TIFFSetField(out, TIFFTAG_BITSPERSAMPLE, info.bitperpixel);

	// Use a cast to uint16 to be safe with LibTIFF's variadic arguments
	TIFFSetField(out, TIFFTAG_ORIENTATION, info.orientation);

	TIFFSetField(out, TIFFTAG_PLANARCONFIG, info.planarconfig);
	TIFFSetField(out, TIFFTAG_PHOTOMETRIC, info.photometric);

	tsize_t linebytes = info.sampleperpixel * info.width;
	unsigned char* buf = NULL;

	if (TIFFScanlineSize(out) < linebytes)
		buf = (unsigned char*)_TIFFmalloc(linebytes);
	else
		buf = (unsigned char*)_TIFFmalloc(TIFFScanlineSize(out));

	TIFFSetField(out, TIFFTAG_ROWSPERSTRIP, TIFFDefaultStripSize(out, info.width * info.sampleperpixel));

	for (uint32_t row = 0; row < info.height; row++)
	{
		memcpy(buf, &(info.data)[row * linebytes], linebytes);
		if (TIFFWriteScanline(out, buf, row, 0) < 0)
			break;
	}

	TIFFClose(out);

	if (buf)
		_TIFFfree(buf);
}