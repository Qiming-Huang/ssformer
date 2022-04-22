
def build(model_name, class_num=1):
############################################   1. CVT
    if model_name == "cvt_pup":
        from .cvt.cvt_pup import cvt_PUP
        model = cvt_PUP(class_num=class_num)
        return model  

    if model_name == "cvt_mla":
        from .cvt.cvt_mla import cvt_mla
        model = cvt_mla(class_num=class_num)
        return model  

    if model_name == "cvt_PPD":
        from .cvt.cvt_PPD import cvt_PPD
        model = cvt_PPD(class_num=class_num)
        return model  
        
    if model_name == "cvt_SD":
        from .cvt.cvt_SD import cvt_SD
        model = cvt_SD(class_num=class_num)
        return model  
        
    if model_name == "cvt_PLD":
        from .cvt.cvt_PLD import cvt_PLD
        model = cvt_PLD(class_num=class_num)
        return model 

#################################################### 2. MIT
    if model_name == "mit_PLD_b2":
        from .mit.mit_PLD_b2 import mit_PLD_b2
        model = mit_PLD_b2(class_num=class_num)
        return model    
          
    if model_name == "mit_PLD_add":
        from .mit.mit_PLD_add import mit_PLD_add
        model = mit_PLD_add(class_num=class_num)
        return model  
        
    if model_name == "mit_mla":
        from .mit.mit_mla import mit_mla
        model = mit_mla(class_num=class_num)
        return model  
        
    if model_name == "mit_pup":
        from .mit.mit_pup import mit_pup
        model = mit_pup(class_num=class_num)
        return model

    if model_name == "mit_SD":
        from .mit.mit_SD import mit_SD
        model = mit_SD(class_num=class_num)
        return model

    if model_name == "mit_PPD":
        from .mit.mit_PPD import mit_PPD
        model = mit_PPD(class_num=class_num)
        return model

    if model_name == "mit_PLD_b4":
        from .mit.mit_PLD_b4 import mit_PLD_b4
        model = mit_PLD_b4(class_num=class_num)
        return model

#################################################### 3. PVT

    if model_name == "pvt_PLD":
        from .pvt.pvt_PLD import pvt_PLD
        model = pvt_PLD(class_num=class_num)
        return model      

    if model_name == "pvt_mla":
        from .pvt.pvt_mla import pvt_mla
        model = pvt_mla(class_num=class_num)
        return model  
        
    if model_name == "pvt_pup":
        from .pvt.pvt_pup import pvt_pup
        model = pvt_pup(class_num=class_num)
        return model

    if model_name == "pvt_SD":
        from .pvt.pvt_SD import pvt_SD
        model = pvt_SD(class_num=class_num)
        return model

    if model_name == "pvt_PPD":
        from .pvt.pvt_PPD import pvt_PPD
        model = pvt_PPD(class_num=class_num)
        return model

#################################################### simVit


    if model_name == "simVit_PLD":
        from .simVit.simVit_PLD import simVit_PLD
        model = simVit_PLD(class_num=class_num)
        return model      

    if model_name == "simVit_mla":
        from .simVit.simVit_mla import simVit_mla
        model = simVit_mla(class_num=class_num)
        return model  
        
    if model_name == "simVit_pup":
        from .simVit.simVit_pup import simVit_pup
        model = simVit_pup(class_num=class_num)
        return model

    if model_name == "simVit_SD":
        from .simVit.simVit_SD import simVit_SD
        model = simVit_SD(class_num=class_num)
        return model

    if model_name == "simVit_PPD":
        from .simVit.simVit_PPD import simVit_PPD
        model = simVit_PPD(class_num=class_num)
        return model

####################################################


    if model_name == "ssa_PLD":
        from .ssa.ssa_PLD import ssa_PLD
        model = ssa_PLD(class_num=class_num)
        return model      

    if model_name == "ssa_mla":
        from .ssa.ssa_mla import ssa_mla
        model = ssa_mla(class_num=class_num)
        return model  
        
    if model_name == "ssa_pup":
        from .ssa.ssa_pup import ssa_pup
        model = ssa_pup(class_num=class_num)
        return model

    if model_name == "ssa_SD":
        from .ssa.ssa_SD import ssa_SD
        model = ssa_SD(class_num=class_num)
        return model

    if model_name == "ssa_PPD":
        from .ssa.ssa_PPD import ssa_PPD
        model = ssa_PPD(class_num=class_num)
        return model

        