# This is a simple wrapper for running the PyKat detector simulation from within gwtoolbox.
# This functionality was originally planned to be part of detectors.py, but issues with import statements executed within a function necessitated me to move them to a separate source file.


###
# default values for LIGO
# const Larm 3995
# const Pin  125
###

try:
    from pykat import finesse
    from pykat.detectors import *
    from pykat.components import *
    from pykat.commands import *
    from pykat.structs import *
except:
    print("Error when trying to import Finesse functions through PyKAT. Check your installation!")
    raise Exception("FINESSE amd/or PyKat are not installed or accessible. Install them, and check if the pykat module is accessible from your python prompt.")

class Session:
    """
    Object capable of running a PyKat simulation according to kat_code.
    
    Parameters:
      pars (list of floats): new setup for detector
    """

    def __init__(self, pars=None):
        """
        The user should supply a valid katCode variable when instantiating this class. Check if that is the case here!
        Parameters:
          pars (list of floats): new setup for detector
        """

        self.kat_code = """
        %------------------------------------------------------------------------
        % Finesse input file to model a Michelson interferometer
        % with power and signal recycling. The setup is based on
        % the aLIGO reference design, DCC number M060056
        % Daniel Brown 17.05.2014
        %------------------------------------------------------------------------
        
        l l1 $Pin 0 nin
        s s1 0 nin nprc1
        # Power recycling mirror
        m1 prm $prmT 37.5u 90 nprc1 nprc2
        s  prc $lprc nprc2 nbsin
        # Central beamsplitter
        bs bs1 .5 .5 0 45 nbsin n0y n0x nbsout
        
        # X-arm
        s ichx $lmichx n0x n1x
        m1 itmx $itmT 37.5u 90 n1x n2x
        s armx $Larm n2x n3x
        m1 etmx 5u 37.5u 89.999875 n3x n4x
        attr itmx mass $Mtm zmech sus1
        attr etmx mass $Mtm zmech sus1
        
        # Y-arm
        s  ichy $lmichy n0y n1y
        m1 itmy $itmT 37.5u $michy_phi n1y n2y
        s  army $Larm n2y n3y
        m1 etmy 5u 37.5u 0.000125 n3y n4y
        attr itmy mass $Mtm zmech sus1
        attr etmy mass $Mtm zmech sus1
        
        # Signal recycling mirror
        s  src $lsrc nbsout nsrc1
        m1 srm $srmT 37.5u $srm_phi nsrc1 nsrc2
        
        # Force-to-position transfer function for longitudinal
        # motions of test masses
        tf sus1 1 0 p $mech_fres $mech_Q
        const mech_fres 1  # 9 sus-thermal spike
        const mech_Q    1M # Guess for suspension Q factor
        # DC readout: 100mW = michy_phi 0.07 _or_ darm_phi .00025
        const michy_phi 0
        const darm_phi  .00025
        
        const Larm    LARM_VALUE
        const Pin     PIN_VALUE
        const itmT    ITMT_VALUE
        const srmT    SRMT_VALUE
        const prmT    PRMT_VALUE
        const Mtm     MTM_VALUE
        const srm_phi SRMPHI_VALUE
        const lmichx  4.5
        const lmichy  4.45
        const lprc    LPRC_VALUE
        const lsrc    LSRC_VALUE
        
        # A squeezed source could be injected into the dark port
        sq sq1 0 0 90 nsrc2
        
        # Differentially modulate the arm lengths
        fsig darm  armx 1 0
        fsig darm2 army 1 180
        
        # Output the full quantum noise limited sensitivity
        qnoisedS NSR_with_RP    1 $fs nsrc2
        # Output just the shot noise limited sensitivity
        # qshotS   NSR_without_RP 1 $fs nsrc2
        
        # We could also display the quantum noise and the signal
        # separately by uncommenting these two lines.
        # qnoised noise 1 $fs nsrc2
        # pd1     signal  $fs nsrc2
        
        xaxis darm f log 5 5k 1000
        yaxis log abs
        
        """
        if pars != None:
            print("Using form-supplied parameter values:", pars,"<br>")
            try:
              for i in range(0, len(pars)):
                # 'pars' is a list where every element has 2 elements of its own: a parameter placeholder name and a value.
                # We go through all entries in the 'pars' variable and substitute the supplied variable values in the relevant spots.
                # Note that both elements, the name and the value, for each parameter are supplied as strings!
                # Example: pars = [["LARM_VALUE", "10000"],["PIN_VALUE", "550"]]
                self.kat_code = self.kat_code.replace(pars[i][0],pars[i][1])
              # Having replaced all the relevant parameter values, we now need to fill in any others that have not been supplied - just in case.
            except Exception as e:
              print("String replacement error encountered!\n")
              print(e)
            else:
              #print("String replacement ok. Kat_code: ", self.kat_code,"\n")
              fred = 0.
        else:
            #raise Exception("No katCode supplied when instantiating the Session object! Please change the instantiation to include a valid code string for the katCode function argument.")
            print("No katCode supplied, so I will use the standard code with the default values...<br>")
        self.kat_code = self.kat_code.replace("LARM_VALUE", "3995")
        self.kat_code = self.kat_code.replace("PIN_VALUE", "125")
        self.kat_code = self.kat_code.replace("ITMT_VALUE", "0.014")
        self.kat_code = self.kat_code.replace("SRMT_VALUE", "0.2")
        self.kat_code = self.kat_code.replace("PRMT_VALUE", "0.03")
        self.kat_code = self.kat_code.replace("MTM_VALUE", "40")
        self.kat_code = self.kat_code.replace("SRMPHI_VALUE", "-90")
        self.kat_code = self.kat_code.replace("LPRC_VALUE", "53")
        self.kat_code = self.kat_code.replace("LSRC_VALUE", "50.525")

    def run(self):
        """
        Run PyKat in this block, using the code specified in the katCode variable.
        """
        print("Running sim session.<br>")
        # Besides the PyKat packages, we need to have valid FINESSE code to run - so check if we got something before running Kat.
        if self.kat_code != None:

            # this kat object represents one single simulation, it contains
            # all the objects and their various states.
            try:
              kat = finesse.kat()
            except:
              print("Cannot instantiate a kat object...<br>")
            else:
              print("Kat object instantiated.<br>")

            # Currently, the kat object is empty. We can fill it using a block
            # string of normal FINESSE commands by parsing them.
            try:
              kat.parse(self.kat_code)
            except:
              print("Could not parse kat code!<br>")
            else:
              print("Kat code parsed.<br>")

            # Once we have some simulation built up we can run it simply by calling...
            try:
              out = kat.run()
            except:
              print("<br>Could not run kat!<br>")
            else:
              print("<br>Ran kat.<br>")

            # Return PyKat's output to the 'detectors' module from which this function was called.
            return out
        else:
            print("No kat code defined!<br>")
            raise Exception("No PyKat code supplied when executing function run() in simulator.py! Make sure to initialize the session with the appropriate KAT code.")

