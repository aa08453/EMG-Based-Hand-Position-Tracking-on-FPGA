// Minimal stand-in for Vitis' hls_stream.h, for C simulation with g++ only (make). Vitis uses its own header.
// Unbounded FIFO: read() on an empty stream is a testbench or design error and aborts.
#ifndef HLS_STREAM_SIM_H
#define HLS_STREAM_SIM_H
#include <cstdio>
#include <cstdlib>
#include <deque>
#include <string>

namespace hls {
template <typename T>
class stream {
public:
    stream() : name_("stream") {}
    explicit stream(const char* name) : name_(name) {}
    void write(const T& v) { q_.push_back(v); }
    T read() {
        if (q_.empty()) { std::fprintf(stderr, "hls::stream '%s': read from an empty stream\n", name_.c_str()); std::abort(); }
        T v = q_.front(); q_.pop_front(); return v;
    }
    void read(T& v) { v = read(); }
    bool read_nb(T& v) { if (q_.empty()) return false; v = read(); return true; }
    bool write_nb(const T& v) { write(v); return true; }
    bool empty() const { return q_.empty(); }
    bool full() const { return false; }
    std::size_t size() const { return q_.size(); }
    stream& operator<<(const T& v) { write(v); return *this; }
    stream& operator>>(T& v) { v = read(); return *this; }
private:
    std::string name_;
    std::deque<T> q_;
};
}  // namespace hls
#endif
