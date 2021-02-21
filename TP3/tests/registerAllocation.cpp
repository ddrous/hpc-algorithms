#include <string>
#include <iostream>
#include <vector>
#include <unordered_map>
#include <set>
#include <cassert>

/* Enumaration of all the possible operation types */
enum InstructionType{
    ASSIGNEMENT,
    OPERATION,
    RET
};

/**
 * A structure that holds and instruction e.g: t3 = t1 + t2
 * @param dest t3
 * @param src1 t1
 * @param src2 t2
 * @param op +
 */
struct Instruction{
    InstructionType type;
    
    std::string dest;
    std::string src1;
    std::string op;
    std::string src2;
    
    void print() const {
        std::cout << " >>";
        switch(type){
        case ASSIGNEMENT:
            std::cout << dest << " = " << src1 << std::endl;
            break;
        case OPERATION:
            std::cout << dest << " = " << src1 << op << src2 << std::endl;
            break;
        case RET:
            std::cout << "return " << dest << std::endl;
            break;
        default:
            std::cout << "Unknown type " << type << " : " << dest << "," << src1 << "," << op << "," << src2 << std::endl;
        }
    }
};

/* A function to genreate a bunch of instructions */
std::vector<Instruction> generateInstructions(){
    std::vector<Instruction> instructions;

    instructions.emplace_back(Instruction{ASSIGNEMENT, "t0", "$0", "", ""});
    instructions.emplace_back(Instruction{ASSIGNEMENT, "t1", "$10", "", ""});
    instructions.emplace_back(Instruction{OPERATION, "t3", "t0", "+", "x"});
    instructions.emplace_back(Instruction{OPERATION, "t4", "t0", "*", "t1"});
    instructions.emplace_back(Instruction{OPERATION, "t5", "t4", "+", "t0"});
    instructions.emplace_back(Instruction{OPERATION, "t6", "y", "*", "t1"});
    instructions.emplace_back(Instruction{OPERATION, "t7", "t6", "*", "t1"});
    instructions.emplace_back(Instruction{OPERATION, "t8", "t6", "*", "t1"});
    instructions.emplace_back(Instruction{OPERATION, "t9", "t0", "*", "t8"});
    instructions.emplace_back(Instruction{OPERATION, "t10", "t4", "*", "t8"});
    instructions.emplace_back(Instruction{OPERATION, "t11", "t4", "*", "t8"});
    instructions.emplace_back(Instruction{OPERATION, "t12", "t3", "*", "t7"});
    instructions.emplace_back(Instruction{OPERATION, "t13", "t12", "*", "t11"});
    instructions.emplace_back(Instruction{OPERATION, "t14", "t12", "*", "t11"});
    instructions.emplace_back(Instruction{RET, "t13", "", "", ""});

    return instructions;
}

/* A function that prints a vector of instructions */
void printAllInstructions(const std::vector<Instruction>& instructions){
    for(const auto &ins : instructions){
        ins.print();
    }
}

/* A function that checks the first character of a myStr */
bool startsWith(const std::string& myStr, const std::string& myChar){
    return myStr.substr(0,1) == myChar;
}

/* Checks if myStr is in myMap */
bool isInMap(const std::string& myStr, const std::unordered_map<std::string, std::string>& myMap){
    return myMap.find(myStr) != myMap.end();;
}

/* Checks if myStr is in mySet */
bool isInSet(const std::string& myStr, const std::set<std::string>& mySet){
    return mySet.find(myStr) != mySet.end();;
}

/**
 * @brief Given and instruction, this fucntion computes the register for allocating 
 * either of its operand (that variable at `src1` or `scr2`). It goes through 
 * all the past register allocations and checks if it has used the variable 
 * before (backward pass). If it is the case, it checks check that that register 
 * has not been used since to store another variable (forward pass). If it has, 
 * the value in the register is no more valid, we just store our variable in 
 * another register.
 * 
 * @param srcType either src1 or src2
 * @param cptRegUse The register that will be used if no optimization is possible
 * @param NbRegistersInCPU The total number of registers avaialable
 * @param iter a reference to the instruction whose src1/scr2 will be allocated
 * @param instructions Contains the intruction and all the past optimized ones
 */
void allocateSrc(std::string srcType, long int& cptRegUse, const long int& NbRegistersInCPU, std::vector<Instruction>::iterator& iter, std::vector<Instruction>& instructions){

    // This register will be used if no otmization is found 
    std::string regToUse = "%r" + std::to_string(cptRegUse);
    // Indicates whether a new instruction will be created 
    bool createInstruction = true;
    
    // Defines the variable in question
    std::string src;
    if (srcType == "src1")
        src = (*iter).src1;
    else if(srcType == "src2")
        src = (*iter).src2;
    else
        std::cout << "Not a valid source!" << std::endl;

    // The very first indtruction can never be optimized
    if(iter != instructions.begin()){

        // See if src has been loaded in a register before (backward pass)                   
        auto bwdIter = iter-1;
        while (bwdIter!=instructions.begin()-1)
        {
            auto ins = *bwdIter;
            
            // if src was a src1 in an assignment, then dest was a register
            if(src == ins.src1){
                regToUse = ins.dest;
                createInstruction = false;
                break;
            }

            // if src was a dest, then the src1 was a register
            if(src == ins.dest){
                regToUse = ins.src1;
                createInstruction = false;
                break;
            }

            --bwdIter;
        }

        // If the variable src has never been used before, we need an allocation
        if(bwdIter == instructions.begin()-1){
            createInstruction = true;
        } 

        // Now that we know a potential register to use, let's check if it has not been used since (forward pass)
        else{
            auto fwdIter = bwdIter;
            while (fwdIter != iter)
            {
                auto ins = *fwdIter;

                // If the register was used as dest, it's no more valid 
                if(regToUse == ins.dest){
                    regToUse = "%r" + std::to_string(cptRegUse);
                    createInstruction = true;
                    break;
                }

                ++fwdIter;
            }
        }
    }

    // Allocate a register for src (no optimization was possible)
    if (createInstruction){
        iter = instructions.insert(iter, Instruction{ASSIGNEMENT, regToUse, src, "", ""}); 
        ++iter;
    }

    // Replace src by the register we have found or allocated for it
    if (srcType == "src1")
        (*iter).src1 = regToUse;
    else if(srcType == "src2")
        (*iter).src2 = regToUse; 
    else
        std::cout << "Not a valid src!" << std::endl;

    // The next instruction will (maybe) use the one after this one
    cptRegUse = (atoi(regToUse.substr(2,3).c_str()) + 1 ) % NbRegistersInCPU ;
}



int main(){

    std::vector<Instruction> instructions = generateInstructions();
    
    std::cout << "Original instructions:" << std::endl;
    printAllInstructions(instructions);
    
    // Constant propagation
    {
        // Use it to know if a variable can be replaced by a formula
        std::unordered_map<std::string, std::string> constValues;
        
        // TODO
        for(auto &ins : instructions){
            // Evaluate the first source term and eventually add it to the map 
            if (isInMap(ins.src1, constValues))
                ins.src1 = constValues[ins.src1];

            // Evaluate the second source term
            if (isInMap(ins.src2, constValues))
                ins.src2 = constValues[ins.src2];

            // An assigment with a constant operand yields a constant
            if (ins.type==ASSIGNEMENT && startsWith(ins.src1, "$"))
                constValues[ins.dest] = ins.src1;

            // An operation with both operands constant becomes an assignment
            if (ins.type==OPERATION && startsWith(ins.src1, "$") && startsWith(ins.src2, "$")){
                ins.type = ASSIGNEMENT;
                ins.src1 = "$("+ins.src1+ins.op+ins.src2+")";
                ins.op = "";
                ins.src2 = "";

                constValues[ins.dest] = ins.src1;
            }
        }
        
        std::cout << "\nAfter constant propagation:" << std::endl;
        printAllInstructions(instructions);
    }

    // Remove unused
    {  
        std::cout << "\nProceed to remove unused variables:" << std::endl;
        // Consider that "used" will store all the variables that are really used
        std::set<std::string> used;
        
        assert(instructions.size() && instructions.back().type == RET);
        
        // TODO fill "used"

        // The returned variable is most certainly used
        used.insert(instructions.back().dest);

        // Loop over the instructions in reverse order
        for(auto iter=instructions.end()-1; iter!=instructions.begin()-1; --iter){
            
            // Just for simplicty
            auto &ins = *iter;

            // Operands src1 and src2 are really used only if the destination is used
            if (isInSet(ins.dest, used)){

                // In ASSIGNMENT or OPERATION, a used operand cannot be a constant
                if (startsWith(ins.src1, "$") == false)
                    used.insert(ins.src1);

                // Only OPERATIONS have a 2nd operand that cannot be a constant
                if ((ins.type == OPERATION) && startsWith(ins.src2, "$") == false)
                    used.insert(ins.src2);
            }
        }
        {
            // We remove all variables not in "used"
            auto iterEnd = instructions.end();
            for(auto iter = instructions.begin() ; iter != iterEnd ;){
                if(used.find((*iter).dest) == used.end()){
                    std::cout << "Erase : " << (*iter).dest << std::endl;
                    instructions.erase(iter);
                    iterEnd = instructions.end();
                }
                else{
                     ++iter;
                }
            }
        }
        
        std::cout << "\nAfter removing unused variables:" << std::endl;
        printAllInstructions(instructions);
    }
        
    // Dumb register allocations
    /* {
        const long int NbRegistersInCPU = 3;
        
        auto iterEnd = instructions.end();
        for(auto iter = instructions.begin() ; iter != iterEnd ;){
            // If this is an operation
            if((*iter).type == OPERATION){
                long int cptRegUse = 0;
                if((*iter).src1.rfind("$",0) != 0){

                    iter = instructions.insert(iter, Instruction{ASSIGNEMENT, "%r" + std::to_string(cptRegUse), (*iter).src1, "", ""});     // On rajoute l'instruction de déplacement de src1 vers un registre                
                    ++iter;
                    (*iter).src1 = "%r" + std::to_string(cptRegUse);
                    cptRegUse += 1;     // The src1 in the original instruction is now the register
                }

                iter = instructions.insert(iter+1, Instruction{ASSIGNEMENT, (*iter).dest, "%r" + std::to_string(cptRegUse), "", ""});   // Next to the operation, the instruction to mode its destination into a pointer is added
                --iter;     // Retour a l'ancienne position
                (*iter).dest = "%r" + std::to_string(cptRegUse);    // The destination is now a register
                cptRegUse += 1;
                iterEnd = instructions.end();
                iter += 2;      // Over to the next high-level instruction
                assert(cptRegUse <= NbRegistersInCPU);
            }
            else{
                 ++iter;
            }
        }

        std::cout << "\nAfter register allocation:" << std::endl;
        printAllInstructions(instructions);
        
    } */

    // My attempt (explained in the function 'allocateSrc()')
    {
        const long int NbRegistersInCPU = 3;
        auto iterEnd = instructions.end();

        for(auto iter = instructions.begin() ; iter != iterEnd ;){

            // If this is an operation
            if((*iter).type == OPERATION){

                // The register to be used for allocations
                long int cptRegUse = 0;

                // If src1 is not a constant, find or allocate a register
                if((*iter).src1.rfind("$",0) != 0)
                    allocateSrc("src1", cptRegUse, NbRegistersInCPU, iter, instructions);

                // The same goes for src2
                if((*iter).src2.rfind("$",0) != 0)
                    allocateSrc("src2", cptRegUse, NbRegistersInCPU, iter, instructions);

                // A new instruction must be created for the dest; it is a value of interest
                iter = instructions.insert(iter+1, Instruction{ASSIGNEMENT, (*iter).dest, "%r" + std::to_string(cptRegUse), "", ""});
                --iter;
                (*iter).dest = "%r" + std::to_string(cptRegUse);
                cptRegUse = (cptRegUse + 1) % NbRegistersInCPU;
                iterEnd = instructions.end();
                iter += 2;
            }
            else{
                 ++iter;
            }
        }

        std::cout << "\nAfter register allocation:" << std::endl;
        printAllInstructions(instructions);

    } 

    return 0;
}

