//
// Created by lars
// Based on the paper Computing Liveness Sets for SSA-Form Programs by Florian Brandner,
// Benoit Boissinot, Alain Darte, Benoit Dupont de Dinechin, Fabrice Rastelloss

#include "LivenessAnalysis.h"

using namespace llvm;
using namespace pacxx;

LivenessAnalyzer::LivenessAnalyzer() : FunctionPass(ID) {
  initializeLivenessAnalyzerPass(*PassRegistry::getPassRegistry());
}

LivenessAnalyzer::~LivenessAnalyzer() {}

void LivenessAnalyzer::releaseMemory() {}

void LivenessAnalyzer::getAnalysisUsage(AnalysisUsage &AU) const {}


bool LivenessAnalyzer::runOnFunction(Function &F) {

  computeLiveSets(F);

  return false;
}

void LivenessAnalyzer::computeLiveSets(Function &F) {

  for (auto BI = F.begin(), BE = F.end(); BI != BE; ++BI) {

    BasicBlock *BB = &*BI;
    set<Use *> phiUses;
    set<BasicBlock *> visitedBlocks;
    getPhiUses(BB, visitedBlocks, phiUses, BB);

    for(auto use : phiUses) {
      _out[BB].insert(use->get());
      upAndMark(BB, use);
    }

    for(auto I = BB->begin(), IE = BB->end(); I != IE; ++I) {
      if(!isa<PHINode>(*I))
        for(auto &op : I->operands()) {
          if(isa<Instruction>(op))
            upAndMark(BB, &op);
        }
    }
  }
}

void LivenessAnalyzer::upAndMark(BasicBlock *BB, Use *use) {

  Value *useValue = use->get();
  if(Instruction *inst = dyn_cast<Instruction>(useValue)) {
    if (!isa<PHINode>(inst))
      if (inst->getParent() == BB) return;
  }

  if(_in[BB].count(useValue) > 0) return;

  _in[BB].insert(useValue);

  set<Value *> phiDefs = getPhiDefs(BB);

  if(phiDefs.count(useValue) > 0) return;

  for(auto I = pred_begin(BB), IE = pred_end(BB); I != IE; ++I) {
    BasicBlock *pred = *I;
    _out[pred].insert(useValue);
    upAndMark(pred, use);
  }
}

void LivenessAnalyzer::getPhiUses(BasicBlock *current,
                                             set<BasicBlock *> &visited,
                                             set<Use *> &uses,
                                             BasicBlock *orig) {

  if(visited.find(current) != visited.end()) return;
  visited.insert(current);

  for (auto I = succ_begin(current), IE = succ_end(current); I != IE; ++I) {
    BasicBlock *succ = *I;
    for(auto BI = succ->begin(), BE = succ->end(); BI != BE; ++BI) {
      // find PHINodes of successors
      if(PHINode *phi = dyn_cast<PHINode>(&*BI)) {
        for(auto &use : phi->incoming_values()) {
          if(phi->getIncomingBlock(use) == orig && isa<Instruction>(use.get()))
            uses.insert(&use);
        }
      }
    }
    //recurse
    getPhiUses(succ, visited, uses, orig);
  }
}


set<Value *> LivenessAnalyzer::getPhiDefs(BasicBlock *BB) {
  set<Value *> uses;
  for(auto I = BB->begin(), IE = BB->end(); I != IE; ++I) {
    if(PHINode *phi = dyn_cast<PHINode>(&*I)) {
      uses.insert(phi);
    }
  }
  return uses;
}

set<Value *> LivenessAnalyzer::getLivingInValuesForBlock(const BasicBlock* block) {
  return _in[block];
}

namespace llvm {
Pass* createPACXXLivenessAnalyzerPass() { return new LivenessAnalyzer(); }
}

string LivenessAnalyzer::toString(map<const BasicBlock *, set<Value *>> &map) {
  string text;
  raw_string_ostream ss(text);

  for(auto elem : map) {
    ss << elem.first->getName() << " : \n";
    ss << toString(elem.second);
    ss << "\n\n";
  }

  return ss.str();
}

string LivenessAnalyzer::toString(set<Value *> &set) {
  string text;
  raw_string_ostream ss(text);

  for(auto val : set) {
    val->print(ss, true);
    ss << "\n";
  }

  return ss.str();
}

char LivenessAnalyzer::ID = 0;

INITIALIZE_PASS_BEGIN(LivenessAnalyzer, "native-liveness", "Liveness Analysis", true, true)
INITIALIZE_PASS_END(LivenessAnalyzer, "native-liveness", "Liveness Analysis", true, true)

