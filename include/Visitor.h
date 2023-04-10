#ifndef VISITOR_H
#define VISITOR_H

#include <ostream>

// Forward declaration
class Layer;

class Visitor
{
public:
    virtual ~Visitor() = default;
    virtual void visit(Layer& layer) = 0;
};

class DescriptionVisitor : public Visitor
{
public:
    explicit DescriptionVisitor(std::ostream& os): out(os)
    {
    }
    void visit(Layer& layer) override
    {
        
    }
private:
    std::ostream& out;
};

#endif