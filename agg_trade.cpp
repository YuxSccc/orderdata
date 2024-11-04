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
namespace fs = std::filesystem;

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

// 获取小数精度
int get_decimal_precision(double number)
{
    std::stringstream ss;
    ss << std::fixed << number;
    std::string str = ss.str();
    size_t decimal_pos = str.find('.');
    if (decimal_pos == std::string::npos)
        return 0;

    // 从后往前找第一个非0数字
    int precision = 0;
    for (int i = str.length() - 1; i > decimal_pos; i--)
    {
        if (str[i] != '0')
        {
            precision = i - decimal_pos;
            break;
        }
    }
    return precision;
}

// 解析CSV行
Trade parse_csv_line(const std::string &line, bool has_header)
{
    std::stringstream ss(line);
    std::string token;
    Trade trade;

    std::getline(ss, token, ',');
    trade.id = std::stoll(token);

    std::getline(ss, token, ',');
    trade.price = std::stod(token);

    std::getline(ss, token, ',');
    trade.qty = std::stod(token);

    std::getline(ss, token, ',');
    trade.quote_qty = std::stod(token);

    std::getline(ss, token, ',');
    trade.time = std::stoll(token);

    std::getline(ss, token, ',');
    trade.is_buyer_maker = (token == "true" || token == "True" || token == "1");

    return trade;
}

#include <thread>
#include <mutex>
#include <semaphore>

int main()
{
    const std::string path_prefix = "/mnt/e/orderdata/binance";
    const int64_t pre_agg_duration = 100; // 聚合时间精度

    std::vector<std::thread> threads;
    std::counting_semaphore<8> sem(8);

    for (const auto &entry : fs::directory_iterator(path_prefix + "/rawdata"))
    {
        auto task = [&](auto entry)
        {
            if (entry.path().extension() != ".csv")
                return;

            std::string filename = entry.path().stem().string();
            std::string output_path = path_prefix + "/agg_trades/" + filename + ".csv";

            // 检查输出文件是否已存在
            if (fs::exists(output_path))
            {
                std::cout << "Skip existing file: " << filename << std::endl;
                return;
            }

            sem.acquire();
            std::cout << "Processing: " << filename << std::endl;

            // 读取并处理文件
            std::ifstream infile(entry.path());
            std::string line;

            // 检查是否有header
            bool has_header = false;
            if (std::getline(infile, line))
            {
                has_header = line.find("id") != std::string::npos;
                infile.seekg(0);
                if (has_header)
                    std::getline(infile, line); // 跳过header
            }

            std::vector<AggTrade> aggregated;
            AggTrade *buy_side_agg = nullptr;
            AggTrade *sell_side_agg = nullptr;
            int max_qty_precision = 0;
            int max_quote_qty_precision = 0;

            size_t line_count = 0;
            while (std::getline(infile, line))
            {
                // if (++line_count % 10000 == 0)
                // {
                //     std::cout << "Processed " << line_count << " lines\r" << std::flush;
                // }

                Trade trade = parse_csv_line(line, has_header);

                // 更新精度
                max_qty_precision = std::max(max_qty_precision, get_decimal_precision(trade.qty));
                max_quote_qty_precision = std::max(max_quote_qty_precision, get_decimal_precision(trade.quote_qty));

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

            // 写入结果
            std::ofstream outfile(output_path);
            outfile << "id,price,qty,quote_qty,time,is_buyer_maker,count\n";

            for (const auto &agg : aggregated)
            {
                outfile << agg.id << ","
                        << std::fixed << std::setprecision(2) << agg.price << ","
                        << std::setprecision(max_qty_precision) << agg.qty << ","
                        << std::setprecision(max_quote_qty_precision) << agg.quote_qty << ","
                        << agg.time << ","
                        << (agg.is_buyer_maker ? "true" : "false") << ","
                        << agg.count << "\n";
            }
            std::cout << "\nCompleted processing " << filename << std::endl;
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
