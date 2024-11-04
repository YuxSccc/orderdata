#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <filesystem>
#include <algorithm>
#include <thread>
#include <semaphore>
#include <Windows.h>
#include <chrono>
#include <complex>

namespace fs = std::filesystem;

struct Trade
{
    int64_t id;
    double price;
    double qty;
    double quote_qty;
    int64_t time;
    bool is_buyer_maker;

    bool operator<(const Trade &other) const
    {
        if (time == other.time)
            return id < other.id;
        return time < other.time;
    }
};

// 快速字符串转数字
inline int64_t fast_atoll(const char *str, char const **end)
{
    int64_t val = 0;
    while (*str >= '0' && *str <= '9')
    {
        val = val * 10 + (*str - '0');
        str++;
    }
    *end = str;
    return val;
}

std::string GetLastErrorAsString()
{
    DWORD errorCode = GetLastError();
    if (errorCode == 0)
    {
        return std::string();
    }

    LPSTR messageBuffer = nullptr;
    size_t size = FormatMessageA(
        FORMAT_MESSAGE_ALLOCATE_BUFFER |
            FORMAT_MESSAGE_FROM_SYSTEM |
            FORMAT_MESSAGE_IGNORE_INSERTS,
        NULL,
        errorCode,
        MAKELANGID(LANG_ENGLISH, SUBLANG_ENGLISH_US), // 使用英语区域设置
        (LPSTR)&messageBuffer,
        0,
        NULL);

    std::string message;
    if (size > 0 && messageBuffer != nullptr)
    {
        message = std::string(messageBuffer, size);
        LocalFree(messageBuffer);

        // 移除末尾的换行符和回车符
        while (!message.empty() && (message.back() == '\n' || message.back() == '\r'))
        {
            message.pop_back();
        }
    }
    else
    {
        message = "Error code: " + std::to_string(errorCode);
    }

    return message;
}

inline double fast_atof(const char *str, char const **end)
{
    double val = 0;
    bool neg = false;
    if (*str == '-')
    {
        neg = true;
        str++;
    }

    while (*str >= '0' && *str <= '9')
    {
        val = val * 10 + (*str - '0');
        str++;
    }

    if (*str == '.')
    {
        str++;
        double factor = 0.1;
        while (*str >= '0' && *str <= '9')
        {
            val += (*str - '0') * factor;
            factor *= 0.1;
            str++;
        }
    }

    *end = str;
    return neg ? -val : val;
}

// 快速解析一行数据
Trade parse_line(char const *start, char const **end)
{
    Trade trade;
    const char *p = start;

    trade.id = fast_atoll(p, &p);
    p++; // skip comma

    trade.price = fast_atof(p, &p);
    p++; // skip comma

    trade.qty = fast_atof(p, &p);
    p++; // skip comma

    trade.quote_qty = fast_atof(p, &p);
    p++; // skip comma

    trade.time = fast_atoll(p, &p);
    p++; // skip comma

    trade.is_buyer_maker = (*p == 't' || *p == 'T' || *p == '1');

    // 找到行尾
    while (*p && *p != '\n' && *p != '\r')
        p++;
    if (*p == '\r' && *(p + 1) == '\n')
        p += 2;
    else if (*p == '\n')
        p++;

    *end = p;
    return trade;
}

bool is_header(const char *line)
{
    return strstr(line, "id,price") != nullptr;
}

void write_file_optimized(const fs::path &path, const std::vector<Trade> &trades)
{
    auto total_start = std::chrono::high_resolution_clock::now();
    auto last_report = total_start;
    size_t total_bytes = 0;

    std::cout << "Starting to process " << trades.size() << " trades" << std::endl;

    // 打开文件
    std::ofstream outfile(path, std::ios::binary);
    const size_t BUFFER_SIZE = 8 * 1024 * 1024; // 8MB
    std::unique_ptr<char[]> buffer(new char[BUFFER_SIZE]);
    outfile.rdbuf()->pubsetbuf(buffer.get(), BUFFER_SIZE);

    // 快速整数转字符串
    auto format_int = [](char *buf, int64_t val) -> char *
    {
        char tmp[32];
        char *p = tmp + sizeof(tmp);
        *--p = '\0';
        bool neg = val < 0;
        uint64_t uval = neg ? -val : val;

        do
        {
            *--p = '0' + (uval % 10);
            uval /= 10;
        } while (uval > 0);

        if (neg)
            *--p = '-';
        size_t len = tmp + sizeof(tmp) - p - 1;
        memcpy(buf, p, len);
        return buf + len;
    };

    // 快速浮点数转字符串
    auto format_double = [&format_int](char *buf, double val, int precision) -> char *
    {
        // 处理整数部分
        int64_t int_part = static_cast<int64_t>(val);
        char *p = format_int(buf, int_part);
        *p++ = '.';

        // 处理小数部分
        double frac_part = val - int_part;
        if (frac_part < 0)
            frac_part = -frac_part;
        int64_t scaled = static_cast<int64_t>(frac_part * std::pow(10, precision) + 0.5);

        // 补齐前导零
        int zeros = precision - 1;
        while (scaled < std::pow(10, zeros) && zeros > 0)
        {
            *p++ = '0';
            zeros--;
        }

        return format_int(p, scaled);
    };

    // 写入header
    const char *header = "id,price,qty,quote_qty,time,is_buyer_maker\n";
    outfile.write(header, strlen(header));

    const size_t BATCH_SIZE = 100000; // 每批10万条记录
    std::string data_buffer;
    data_buffer.resize(BATCH_SIZE * 150); // 预分配足够空间，估计每行最多150字节

    size_t processed_trades = 0;
    char *write_pos = data_buffer.data();
    size_t buffer_used = 0;

    // 批量处理
    for (size_t i = 0; i < trades.size(); i += BATCH_SIZE)
    {
        auto format_start = std::chrono::high_resolution_clock::now();
        const size_t end = std::min(i + BATCH_SIZE, trades.size());
        write_pos = data_buffer.data();

        // 格式化数据
        for (size_t j = i; j < end; ++j)
        {
            const auto &trade = trades[j];

            // ID
            write_pos = format_int(write_pos, trade.id);
            *write_pos++ = ',';

            // Price (2位小数)
            write_pos = format_double(write_pos, trade.price, 2);
            *write_pos++ = ',';

            // Qty (8位小数)
            write_pos = format_double(write_pos, trade.qty, 3);
            *write_pos++ = ',';

            // Quote Qty (8位小数)
            write_pos = format_double(write_pos, trade.quote_qty, 4);
            *write_pos++ = ',';

            // Time
            write_pos = format_int(write_pos, trade.time);
            *write_pos++ = ',';

            // Is Buyer Maker
            if (trade.is_buyer_maker)
            {
                memcpy(write_pos, "true\n", 5);
                write_pos += 5;
            }
            else
            {
                memcpy(write_pos, "false\n", 6);
                write_pos += 6;
            }
        }

        auto format_end = std::chrono::high_resolution_clock::now();
        buffer_used = write_pos - data_buffer.data();

        // 写入数据
        auto write_start = std::chrono::high_resolution_clock::now();
        outfile.write(data_buffer.data(), buffer_used);
        auto write_end = std::chrono::high_resolution_clock::now();

        // 更新统计信息
        processed_trades += (end - i);
        total_bytes += buffer_used;

        //        // 每秒报告一次性能
        //        auto now = std::chrono::high_resolution_clock::now();
        //        auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(now - last_report).count();
        //        if (elapsed >= 1000) {
        //            double seconds = elapsed / 1000.0;
        //            double mb_written = total_bytes / (1024.0 * 1024.0);
        //            double format_time = std::chrono::duration_cast<std::chrono::microseconds>(format_end - format_start).count() / 1000.0;
        //            double write_time = std::chrono::duration_cast<std::chrono::microseconds>(write_end - write_start).count() / 1000.0;
        //
        //            std::cout << std::fixed << std::setprecision(2)
        //                      << "Progress: " << (processed_trades * 100.0 / trades.size()) << "% "
        //                      << "(" << processed_trades << "/" << trades.size() << " trades), "
        //                      << "Speed: " << (mb_written / seconds) << " MB/s\n"
        //                      << "Last batch: Format time: " << format_time << "ms, "
        //                      << "Write time: " << write_time << "ms, "
        //                      << "Batch size: " << (buffer_used / 1024.0) << "KB"
        //                      << std::endl;
        //
        //            last_report = now;
        //            total_bytes = 0;
        //        }
    }

    auto total_end = std::chrono::high_resolution_clock::now();
    auto total_time = std::chrono::duration_cast<std::chrono::milliseconds>(total_end - total_start).count();

    std::cout << "\nComplete!\n"
              << "Total time: " << total_time << "ms\n"
              << "Average speed: " << (processed_trades * 1000.0 / total_time) << " trades/second"
              << std::endl;
}

void process_file(const fs::path &input_path, const fs::path &output_dir, std::counting_semaphore<6> &sem)
{
    std::string filename = input_path.stem().string();
    fs::path output_path = output_dir / (filename + "_sorted.csv");

    if (fs::exists(output_path))
    {
        std::cout << "Skip existing file: " << filename << std::endl;
        return;
    }

    sem.acquire();
    std::cout << "Processing: " << filename << std::endl;

    try
    {
        // 使用Windows API打开文件
        HANDLE hFile = CreateFileW(input_path.wstring().c_str(),
                                   GENERIC_READ,
                                   FILE_SHARE_READ,
                                   NULL,
                                   OPEN_EXISTING,
                                   FILE_ATTRIBUTE_NORMAL,
                                   NULL);

        if (hFile == INVALID_HANDLE_VALUE)
        {
            throw std::runtime_error("Failed to open file");
        }

        // 获取文件大小
        LARGE_INTEGER fileSize;
        if (!GetFileSizeEx(hFile, &fileSize))
        {
            CloseHandle(hFile);
            throw std::runtime_error("Failed to get file size");
        }

        // 创建文件映射
        HANDLE hMapping = CreateFileMappingW(hFile,
                                             NULL,
                                             PAGE_READONLY,
                                             fileSize.HighPart,
                                             fileSize.LowPart,
                                             NULL);

        if (hMapping == NULL)
        {
            CloseHandle(hFile);
            throw std::runtime_error("Failed to create file mapping");
        }

        // 映射视图
        char *addr = static_cast<char *>(MapViewOfFile(hMapping,
                                                       FILE_MAP_READ,
                                                       0,
                                                       0,
                                                       fileSize.QuadPart));

        if (addr == nullptr)
        {
            CloseHandle(hMapping);
            CloseHandle(hFile);
            throw std::runtime_error("Failed to map view of file");
        }

        // 预分配vector
        std::vector<Trade> trades;
        trades.reserve(fileSize.QuadPart / 50); // 估计每行平均50字节

        char const *p = addr;
        char const *end = addr + fileSize.QuadPart;

        // 检查header
        if (is_header(p))
        {
            while (*p && *p != '\n' && *p != '\r')
                p++;
            if (*p == '\r' && *(p + 1) == '\n')
                p += 2;
            else if (*p == '\n')
                p++;
        }

        // 快速解析数据
        size_t line_count = 0;
        auto start_time = std::chrono::high_resolution_clock::now();

        while (p < end)
        {
            trades.push_back(parse_line(p, &p));
            //            if ((++line_count & 0xFFFFFF) == 0) { // 每262144行输出一次
            //                auto current_time = std::chrono::high_resolution_clock::now();
            //                auto duration = std::chrono::duration_cast<std::chrono::seconds>(current_time - start_time).count();
            //                if (duration > 0) {
            //                    std::cout << filename << ": " << line_count << " lines, "
            //                              << (line_count / duration) << " lines/sec\r" << std::flush;
            //                }
            //            }
        }

        // 清理映射
        UnmapViewOfFile(addr);
        CloseHandle(hMapping);
        CloseHandle(hFile);

        // 排序
        std::sort(trades.begin(), trades.end());

        auto write_start = std::chrono::high_resolution_clock::now();
        write_file_optimized(output_path, trades);
        auto write_end = std::chrono::high_resolution_clock::now();

        auto write_duration = std::chrono::duration_cast<std::chrono::milliseconds>(write_end - write_start).count();
        std::cout << "Write completed in " << write_duration << "ms ("
                  << (trades.size() * 1000.0 / write_duration) << " lines/sec)" << std::endl;
    }
    catch (const std::exception &e)
    {
        std::cerr << "Error processing " << filename << ": " << e.what() << std::endl;
    }

    sem.release();
}

int main()
{
    const fs::path input_dir = R"(E:\orderdata\binance\unsorted_rawdata)";
    const fs::path output_dir = R"(E:\orderdata\binance\rawdata)";
    SetConsoleOutputCP(CP_UTF8);
    // 创建输出目录
    fs::create_directories(output_dir);

    // 并行处理文件
    std::vector<std::thread> threads;
    std::counting_semaphore<6> sem(6); // 最多8个并行线程

    for (const auto &entry : fs::directory_iterator(input_dir))
    {
        if (entry.path().extension() != ".csv")
            continue;

        threads.emplace_back([&sem, entry, &output_dir]()
                             { process_file(entry.path(), output_dir, sem); });
    }

    // 等待所有线程完成
    for (auto &t : threads)
    {
        t.join();
    }

    std::cout << "All files processed successfully!" << std::endl;
    return 0;
}