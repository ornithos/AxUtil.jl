module Log

using Logging  # Base version!
import Logging: shouldlog, min_enabled_level, catch_exceptions, handle_message
using Dates

#-------------------------------------------------------------------------------
# TimestampLogger
"""
    TimestampLogger(stream=stderr, min_level=Info)
Trivial extension of SimpleLogger (def: logging all messages with level greater than or equal to
`min_level` to `stream`) to include a timestamp.
"""
struct TimestampLogger <: AbstractLogger
    stream::IO
    min_level::LogLevel
    message_limits::Dict{Any,Int}
end
TimestampLogger(stream::IO=stderr, level=Info) = TimestampLogger(stream, level, Dict{Any,Int}())

shouldlog(logger::TimestampLogger, level, _module, group, id) =
    get(logger.message_limits, id, 1) > 0

min_enabled_level(logger::TimestampLogger) = logger.min_level

catch_exceptions(logger::TimestampLogger) = false

function handle_message(logger::TimestampLogger, level, message, _module, group, id,
                        filepath, line; maxlog=nothing, kwargs...)
    if maxlog != nothing && maxlog isa Integer
        remaining = get!(logger.message_limits, id, maxlog)
        logger.message_limits[id] = remaining - 1
        remaining > 0 || return
    end
    buf = IOBuffer()
    iob = IOContext(buf, logger.stream)
    levelstr = level == Logging.Warn ? "Warning" : string(level)
    msglines = split(chomp(string(message)), '\n')
    println(iob, "┌ ", msglines[1])
    for i in 2:length(msglines)
        println(iob, "│ ", msglines[i])
    end
    for (key, val) in kwargs
        println(iob, "│   ", key, " = ", val)
    end
    println(iob, "└ ", " "^8, " @ ", something(_module, "nothing"), " ",
            something(filepath, "nothing"), ":", something(line, "nothing"), " [", levelstr, "] ", Dates.format(now(), "u-dd HH:MM:SS"))
    write(logger.stream, take!(buf))
    nothing
end


end # Logging