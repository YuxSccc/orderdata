#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <filesystem>
#include <algorithm>
#include <thread>
#include <semaphore>

namespace fs = std::filesystem;

struct Trade
{
    int64_t id;
    double price;
    double qty;
    double quote_qty;
    int64_t time;
    bool is_buyer_maker;

    // 用于排序的比较运算符
    bool operator<(const Trade &other) const
    {
        if (time == other.time)
        {
            return id < other.id;
        }
        return time < other.time;
    }
};

// 解析CSV行
Trade parse_csv_line(const std::string &line)
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

bool is_header(const std::string &line)
{
    return line.find("id,price") != std::string::npos;
}

void process_file(const fs::path &input_path, const fs::path &output_dir, std::counting_semaphore<6> &sem)
{
    std::string filename = input_path.stem().string();
    fs::path output_path = output_dir / (filename + "_sorted.csv");

    // 检查输出文件是否已存在
    if (fs::exists(output_path))
    {
        std::cout << "Skip existing file: " << filename << std::endl;
        return;
    }

    sem.acquire();
    std::cout << "Processing: " << filename << std::endl;

    try
    {
        // 读取文件
        std::vector<Trade> trades;
        std::ifstream infile(input_path);
        std::string line;
        bool has_header = false;

        // 检查第一行是否为标题
        if (std::getline(infile, line))
        {
            has_header = is_header(line);
            if (!has_header)
            {
                trades.push_back(parse_csv_line(line));
            }
        }

        // 读取剩余行
        while (std::getline(infile, line))
        {
            trades.push_back(parse_csv_line(line));
        }

        // 按时间戳和ID排序
        std::sort(trades.begin(), trades.end());

        // 写入排序后的数据
        std::ofstream outfile(output_path);
        outfile << "id,price,qty,quote_qty,time,is_buyer_maker\n";

        for (const auto &trade : trades)
        {
            outfile << trade.id << ","
                    << std::fixed << std::setprecision(8) << trade.price << ","
                    << trade.qty << ","
                    << trade.quote_qty << ","
                    << trade.time << ","
                    << (trade.is_buyer_maker ? "true" : "false") << "\n";
        }

        std::cout << "Completed processing " << filename
                  << " (" << trades.size() << " trades)" << std::endl;
    }
    catch (const std::exception &e)
    {
        std::cerr << "Error processing " << filename << ": " << e.what() << std::endl;
    }

    sem.release();
}

int main()
{
    const fs::path input_dir = "/mnt/e/orderdata/binance/unsorted_rawdata";
    const fs::path output_dir = "/mnt/e/orderdata/binance/rawdata";

    // 创建输出目录
    fs::create_directories(output_dir);

    // 并行处理文件
    std::vector<std::thread> threads;
    std::counting_semaphore<6> sem(6);

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