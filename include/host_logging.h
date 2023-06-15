/*
    This file is part of the CAD-ASTRA distribution (git@github.com:cad-astra/cad-astra.git).
    Copyright (c) 2021-2023 imec-Vision Lab, University of Antwerp.

    This program is free software: you can redistribute it and/or modify  
    it under the terms of the GNU General Public License as published by  
    the Free Software Foundation, version 3.

    This program is distributed in the hope that it will be useful, but
    WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
    General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program. If not, see <http://www.gnu.org/licenses/>.
*/

#ifndef HOST_LOGGING_H
#define HOST_LOGGING_H

#include <fstream>
#include <sstream>
#include <string>
#include <iostream>
#include <iomanip>
#include <map>
#include <chrono>

#include "../include/global_config.h"
#include "../include/singleton.h"

#if __DEBUG_PRINTOUT__
#define DEBUG_LOG( MODULE, MSG, ... )                                       \
    do                                                                      \
    {                                                                       \
        CHostLogger::getSingletonPtr()->debug(MODULE, MSG, ## __VA_ARGS__); \
    } while(0)
#else
#define DEBUG_LOG(...)
#endif

#define ADD_SRC_NAME(MSG, ...) __FILE__, MSG, ## __VA_ARGS__
// #define INFO( MSG, ...) static void info(__FILE__, MSG, ## __VA_ARGS__)


namespace MeshFP
{
    enum LogLevel {DISABLE=0, FATAL=1, ERROR=2, WARNING=3, INFO=4, DEBUG=5};
    constexpr unsigned int LOG_FMT_NONE=0x0;
    constexpr unsigned int LOG_FMT_DATE_TIME=0x00001;
    constexpr unsigned int LOG_FMT_LEVEL=LOG_FMT_DATE_TIME<<1;
    constexpr unsigned int LOG_FMT_MODULE =LOG_FMT_LEVEL<<1;
    constexpr unsigned int LOG_FMT_ALL=LOG_FMT_DATE_TIME|LOG_FMT_LEVEL|LOG_FMT_MODULE;

    template<typename... Args>
    inline std::string format(const char *MSG, Args... args)
    {
        char log_buf[LOG_BUFFER_SIZE];
        snprintf(&log_buf[0], LOG_BUFFER_SIZE, MSG, args...);   // Returns the number of chars written
        std::string s(log_buf);
        // std::cerr << "log string size: " << s.size() << std::endl;
        return s;
    }
}

extern std::map<MeshFP::LogLevel, std::string> log_level_str;

class CFormatter
{
    public:
        CFormatter() : m_fmt(MeshFP::LOG_FMT_ALL) {}
        void configure(unsigned int fmt)
        {
            m_fmt = fmt;
        }
        std::string get_string(MeshFP::LogLevel level, const std::string &filename, const std::string &message) const
        {
            std::stringstream log_record_ss;
            if(m_fmt&MeshFP::LOG_FMT_DATE_TIME)
            {
                auto tp_now = std::chrono::system_clock::now();
                std::time_t tt_now = std::chrono::system_clock::to_time_t(tp_now);
                std::tm local_tm = *std::localtime(&tt_now);
                log_record_ss  << "["
                               << std::setfill('0') << std::setw(4) << local_tm.tm_year +1900 << "-"
                               << std::setfill('0') << std::setw(2) << local_tm.tm_mon + 1    << "-"
                               << std::setfill('0') << std::setw(2) << local_tm.tm_mday       << " "
                               << std::setfill('0') << std::setw(2) << local_tm.tm_hour       << ":"
                               << std::setfill('0') << std::setw(2) << local_tm.tm_min        << ":"
                               << std::setfill('0') << std::setw(2) << local_tm.tm_sec        << "]";
                // Add a delimiter:
                log_record_ss << " - ";
            }
            if(m_fmt&MeshFP::LOG_FMT_LEVEL)
            {                
                log_record_ss <<  log_level_str[level] << " - ";
            }
            if(m_fmt&MeshFP::LOG_FMT_MODULE)
            {
                // TODO: log filename
                log_record_ss << filename << " - ";
            }
            log_record_ss << message;
            return log_record_ss.str();
        }
    private:
        unsigned int m_fmt;
};

class CHandler
{
    public:
        virtual ~CHandler() {}
        virtual void log(const CFormatter& formatter, MeshFP::LogLevel level, const std::string &filename, const std::string &message) = 0;
};

class CErrStreamHandler : public CHandler
{
    public:
        CErrStreamHandler() : CHandler() {}
        void log(const CFormatter& formatter, MeshFP::LogLevel level, const std::string &filename, const std::string &message) override
        {
            std::cerr << formatter.get_string(level, filename, message) << "\n";
        }
};

class CFileHandler : public CHandler
{
    public:
        CFileHandler(const std::string& filename)
        {
            // TODO: Try - catch!
            m_fs.open(filename, std::fstream::app);
        }
        ~CFileHandler() {m_fs.close();}
        void log(const CFormatter& formatter, MeshFP::LogLevel level, const std::string &filename, const std::string &message) override
        {
            if(m_fs.is_open())
                m_fs << formatter.get_string(level, filename, message) << "\n";
        }
    private:
        std::fstream m_fs;
};

class CHostLogger: public CSingleton<CHostLogger>
{
public:
    CHostLogger() : m_level(MeshFP::WARNING)
    {
        // std::cerr << "Created Logger...\n";
        m_default_handler = new CErrStreamHandler;
        m_handler = m_default_handler;
    }
    ~CHostLogger()
    {
        delete m_default_handler;
    }
    void set_formatter(const CFormatter& formatter) { m_formatter = formatter; }
    CFormatter& formatter() { return m_formatter; }
    void set_handler(CHandler *handler) { m_handler = handler; }
    void set_level(MeshFP::LogLevel level) { m_level= level; }
    template<typename... Args>
    void fatal(const std::string &module, const char*MSG, Args... args)   { log(MeshFP::FATAL,   module, MSG, args...); }
    template<typename... Args>
    void error(const std::string &module, const char*MSG, Args... args)   { log(MeshFP::ERROR,   module, MSG, args...); }
    template<typename... Args>
    void warning(const std::string &module, const char*MSG, Args... args) { log(MeshFP::WARNING, module, MSG, args...); }
    template<typename... Args>
    void info(const std::string &module, const char*MSG, Args... args)    { log(MeshFP::INFO,    module, MSG, args...); }
    template<typename... Args>
    void debug(const std::string &module, const char*MSG, Args... args)   { log(MeshFP::DEBUG,   module, MSG, args...); }
private:
    template<typename... Args>
    void log(MeshFP::LogLevel level, const std::string &module, const char*MSG, Args... args)
    {
        if(m_level >= level)
        {
            m_handler->log(m_formatter, level, module, MeshFP::format(MSG, args...));
        }
    }
    /* data */
    MeshFP::LogLevel m_level;
    CFormatter m_formatter;
    CHandler *m_handler;
    CHandler *m_default_handler;
};

namespace MeshFP
{
    class Logging
    {
    public:
        static void fatal(const std::string &module, const char*msg)
        {
            CHostLogger::getSingletonPtr()->fatal(module, "%s", msg);
        }
        template<typename... Args>
        static void fatal(const std::string &module, const char*msg, Args... args)
        {
            CHostLogger::getSingletonPtr()->fatal(module, msg, args...);
        }
        static void error(const std::string &module, const char*msg)
        {
            CHostLogger::getSingletonPtr()->error(module, "%s", msg);
        }
        template<typename... Args>
        static void error(const std::string &module, const char*msg, Args... args)
        {
            CHostLogger::getSingletonPtr()->error(module, msg, args...);
        }
        static void warning(const std::string &module, const char*msg)
        {
            CHostLogger::getSingletonPtr()->warning(module, "%s", msg);
        }
        template<typename... Args>
        static void warning(const std::string &module, const char*msg, Args... args)
        {
            CHostLogger::getSingletonPtr()->warning(module, msg, args...);
        }
        static void info(const std::string &module, const char*msg)
        {
            CHostLogger::getSingletonPtr()->info(module, "%s", msg);
        }
        template<typename... Args>
        static void info(const std::string &module, const char*msg, Args... args)
        {
            CHostLogger::getSingletonPtr()->info(module, msg, args...);
        }
        static void debug(const std::string &module, const char*msg)
        {
            DEBUG_LOG(module, "%s", msg);
        }
        template<typename... Args>
        static void debug(const std::string &module, const char*msg, Args... args)
        {
            DEBUG_LOG(module, msg, args...);
        }
        static void set_level(MeshFP::LogLevel level)
        {
            CHostLogger::getSingletonPtr()->set_level(level);
        }
        static void configure_format(unsigned int fmt)
        {
            CHostLogger::getSingletonPtr()->formatter().configure(fmt);
        }
        static void set_handler(CHandler *handler)
        {
            CHostLogger::getSingletonPtr()->set_handler(handler);
        }
    };
}
#endif