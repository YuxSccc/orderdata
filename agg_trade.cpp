#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <unordered_map>
#include <filesystem>
#include <sstream>
#include <iomanip>
#include <cmath>
#include <algorithm>
#include <thread>
#include <cstring>
#include <chrono>
namespace fs = std::filesystem;

double eps = 1e-7;

// 交易记录结构
struct Trade
{
    int64_t id;
    double price;
    double qty;
    double quote_qty;
    int64_t time;
    bool is_buyer_maker;
};

// 聚合后的交易记录
struct AggTrade
{
    int64_t id;
    double price;
    double qty;
    double quote_qty;
    int64_t time;
    bool is_buyer_maker;
    int count;
};

struct ProcessingStats
{
    size_t total_trades = 0;
    size_t aggregated_trades = 0;
    double parse_time_ms = 0;
    double write_time_ms = 0;
};

inline int get_decimal_precision(double value)
{
    if (value == 0.0)
        return 0;

    // 移除整数部分
    value = std::abs(value);
    value -= std::floor(value);

    if (value == 0.0)
        return 0;

    // 计算小数位数
    int precision = 0;
    while (value < 0.9999999 && precision < 10)
    { // 使用容差处理浮点误差
        value *= 10;
        value -= std::floor(value);
        precision++;
    }

    return precision;
}

// 快速字符串转数字函数
inline int64_t fast_atoll(const char *str, char const **end)
{
    int64_t val = 0;
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
    *end = str;
    return neg ? -val : val;
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
    val += eps;
    *end = str;
    return neg ? -val : val;
}

// 快速解析一行数据
Trade parse_line(const char *start, const char **end)
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

    trade.is_buyer_maker = (p[0] == 't' || p[0] == 'T' || p[0] == '1');
    while (*p && *p != '\n')
        p++;
    if (*p == '\n')
        p++;

    *end = p;
    return trade;
}

// 快速格式化输出
void fast_write_file(const fs::path &path, const std::vector<AggTrade> &trades,
                     int max_qty_precision, int max_quote_qty_precision, int price_precision)
{
    std::ofstream outfile(path, std::ios::binary);
    const size_t BUFFER_SIZE = 8 * 1024 * 1024; // 8MB buffer
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
        int64_t int_part = static_cast<int64_t>(val);
        char *p = format_int(buf, int_part);
        *p++ = '.';

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
    outfile << "id,price,qty,quote_qty,time,is_buyer_maker,count\n";

    const size_t BATCH_SIZE = 100000;
    std::string data_buffer;
    data_buffer.resize(BATCH_SIZE * 150); // 预分配足够空间

    char *write_pos = data_buffer.data();
    size_t buffer_used = 0;

    for (size_t i = 0; i < trades.size(); i += BATCH_SIZE)
    {
        const size_t end = std::min(i + BATCH_SIZE, trades.size());
        write_pos = data_buffer.data();

        for (size_t j = i; j < end; ++j)
        {
            const auto &trade = trades[j];

            // ID
            write_pos = format_int(write_pos, trade.id);
            *write_pos++ = ',';

            // Price
            write_pos = format_double(write_pos, trade.price, price_precision);
            *write_pos++ = ',';

            // Qty
            write_pos = format_double(write_pos, trade.qty, max_qty_precision);
            *write_pos++ = ',';

            // Quote Qty
            write_pos = format_double(write_pos, trade.quote_qty, max_quote_qty_precision);
            *write_pos++ = ',';

            // Time
            write_pos = format_int(write_pos, trade.time);
            *write_pos++ = ',';

            // Is Buyer Maker
            if (trade.is_buyer_maker)
            {
                memcpy(write_pos, "true", 4);
                write_pos += 4;
            }
            else
            {
                memcpy(write_pos, "false", 5);
                write_pos += 5;
            }
            *write_pos++ = ',';

            // Count
            write_pos = format_int(write_pos, trade.count);
            *write_pos++ = '\n';
        }

        buffer_used = write_pos - data_buffer.data();
        outfile.write(data_buffer.data(), buffer_used);
    }
}

int main()
{
    const std::string intput_dir = R"(E:\orderdata\binance\rawdata)";
    const std::string output_dir = R"(E:\orderdata\binance\agg_trade\)";
    const int64_t pre_agg_duration = 100; // 聚合时间精度

    std::vector<std::thread> threads;
    std::counting_semaphore<6> sem(6);

    int max_qty_precision = 3;
    int max_quote_qty_precision = 4;
    int price_precision = 1;

    for (const auto &entry : fs::directory_iterator(intput_dir))
    {
        auto task = [&](auto entry)
        {
            if (entry.path().extension() != ".csv")
                return;

            std::string filename = entry.path().stem().string();
            std::string output_path = output_dir + filename + ".csv";

            // 检查输出文件是否已存在
            if (fs::exists(output_path))
            {
                std::cout << "Skip existing file: " << filename << std::endl;
                return;
            }

            sem.acquire();
            std::cout << "Processing: " << filename << std::endl;

            ProcessingStats stats;
            auto start_time = std::chrono::high_resolution_clock::now();

            // 读取文件内容
            std::ifstream infile(entry.path(), std::ios::binary);
            if (!infile)
            {
                throw std::runtime_error("Failed to open file");
            }

            const size_t BUFFER_SIZE = 512 * 1024 * 1024; // 512MB buffer
            std::vector<char> buffer(BUFFER_SIZE);
            std::string content;
            content.reserve(BUFFER_SIZE * 2); // 预留双倍空间用于处理跨缓冲区的行
            bool eof = false;
            const size_t MIN_REMAINING = 10000; // 最小剩余长度阈值

            auto parse_start = std::chrono::high_resolution_clock::now();
            std::vector<AggTrade> aggregated;
            AggTrade *buy_side_agg = nullptr;
            AggTrade *sell_side_agg = nullptr;

            // 第一次读取，处理header
            if (infile.read(buffer.data(), BUFFER_SIZE) || infile.gcount() > 0)
            {
                size_t read_size = infile.gcount();
                content.append(buffer.data(), read_size);

                const char *p = content.data();
                const char *end = p + content.size();

                // 检查header
                if (strstr(p, "id,price") != nullptr)
                {
                    while (p < end && *p && *p != '\n')
                        p++;
                    if (p < end && *p == '\n')
                        p++;
                }

                // 主处理循环
                while (true)
                {
                    // 检查是否还有足够的数据来处理
                    if (p >= end)
                    {
                        break;
                    }

                    // 当剩余未处理数据较少时，尝试读取更多数据
                    size_t remaining = end - p;
                    if (!eof && remaining < MIN_REMAINING)
                    {
                        std::string new_content;
                        new_content.reserve(BUFFER_SIZE * 2);

                        // 保存剩余的未处理数据
                        if (remaining > 0)
                        {
                            new_content.assign(p, remaining);
                        }

                        // 读取新数据
                        if (infile.read(buffer.data(), BUFFER_SIZE) || infile.gcount() > 0)
                        {
                            size_t read_size = infile.gcount();
                            new_content.append(buffer.data(), read_size);

                            if (read_size < BUFFER_SIZE)
                            {
                                eof = true;
                            }
                        }
                        else
                        {
                            eof = true;
                        }

                        // 交换新旧内容
                        content = std::move(new_content);

                        // 更新指针
                        p = content.data();
                        end = p + content.size();

                        // 如果没有数据了，退出循环
                        if (content.empty())
                        {
                            break;
                        }
                    }

                    // 确保有足够的数据来解析一行
                    if (p >= end)
                    {
                        break;
                    }

                    // 处理一条交易记录
                    const char *next_p = p;
                    Trade trade = parse_line(p, &next_p);

                    // 确保指针确实前进了
                    if (next_p <= p || next_p > end)
                    {
                        break; // 防止无限循环或越界
                    }
                    p = next_p;

                    stats.total_trades++;
                    // 计算时间戳
                    int64_t ts = (trade.time / pre_agg_duration) * pre_agg_duration;

                    AggTrade **op_agg = trade.is_buyer_maker ? &buy_side_agg : &sell_side_agg;

                    // 如果存在聚合记录但时间不同，保存并重置
                    if (*op_agg && (*op_agg)->time != ts)
                    {
                        aggregated.push_back(**op_agg);
                        delete *op_agg;
                        *op_agg = nullptr;
                    }

                    // 创建新的聚合记录或更新现有记录
                    if (*op_agg == nullptr)
                    {
                        *op_agg = new AggTrade{
                            trade.id,
                            trade.price,
                            trade.qty,
                            trade.quote_qty,
                            ts,
                            trade.is_buyer_maker,
                            1};
                    }
                    else
                    {
                        (*op_agg)->qty += trade.qty;
                        (*op_agg)->quote_qty += trade.quote_qty;
                        (*op_agg)->count++;
                    }
                }
            }

            auto parse_end = std::chrono::high_resolution_clock::now();
            stats.parse_time_ms = std::chrono::duration<double, std::milli>(parse_end - parse_start).count();
            stats.aggregated_trades = aggregated.size();

            // 添加最后的聚合记录
            if (buy_side_agg)
            {
                aggregated.push_back(*buy_side_agg);
                delete buy_side_agg;
            }
            if (sell_side_agg)
            {
                aggregated.push_back(*sell_side_agg);
                delete sell_side_agg;
            }

            // 按时间和ID排序
            std::sort(aggregated.begin(), aggregated.end(),
                      [](const AggTrade &a, const AggTrade &b)
                      {
                          if (a.time == b.time)
                              return a.id < b.id;
                          return a.time < b.time;
                      });

            auto write_start = std::chrono::high_resolution_clock::now();
            fast_write_file(output_path, aggregated, max_qty_precision, max_quote_qty_precision, price_precision);
            auto write_end = std::chrono::high_resolution_clock::now();
            stats.write_time_ms = std::chrono::duration<double, std::milli>(write_end - write_start).count();

            std::cout << "\nCompleted processing " << filename << "\n"
                      << "Total trades: " << stats.total_trades << "\n"
                      << "Aggregated trades: " << stats.aggregated_trades << "\n"
                      << "Parse time: " << std::fixed << std::setprecision(2) << stats.parse_time_ms << "ms\n"
                      << "Write time: " << stats.write_time_ms << "ms\n"
                      << "Total time: " << (stats.parse_time_ms + stats.write_time_ms) << "ms\n"
                      << std::endl;

            sem.release();
        };
        threads.push_back(std::thread(task, entry));
    }
    for (auto &t : threads)
    {
        t.join();
    }
    return 0;
}
