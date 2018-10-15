macro noopwhen(condition, expression)
    quote
        if !($condition)
            $expression
        end
    end |> esc
end