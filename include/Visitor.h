#ifndef VISITOR_H
#define VISITOR_H

#include <ostream>
#include <string>

// Forward declaration
class Layer;

class Visitor
{
public:
    virtual ~Visitor() = default;
    virtual void visit(BaseModule& layer) = 0;
};

class DescriptionVisitor : public Visitor
{
public:
    explicit DescriptionVisitor(std::ostream& os): out(os)
    {
    }
    void visit(BaseModule& layer) override
    {
        out << layer.getName(); 
    }
private:
    std::ostream& out;
};

#endif