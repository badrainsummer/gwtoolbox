# This is a simple wrapper for running the PyKat detector simulation from within gwtoolbox.
# This functionality was originally planned to be part of detectors.py, but issues with import statements executed within a function necessitated me to move them to a separate source file.


###
# default values for LIGO
# var Larm 3995
# var Pin  125
###

try:
    import finesse
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
        ###########################################################################
        ###   Variables
        ###########################################################################
        var Larm LARM_VALUE
        var Pin PIN_VALUE
        var Mtm  MTM_VALUE
        var itmT ITMT_VALUE
        var srmT SRMT_VALUE
        var prmT PRMT_VALUE
        var srm_phi SRMPHI_VALUE
        var lprc LPRC_VALUE
        var lsrc LSRC_VALUE
        var lmichx 4.5
        var lmichy 4.45

        ###########################################################################
        ###   Input optics
        ###########################################################################
        l L0 Pin
        s l_in L0.p1 prm.p1
        # Power recycling mirror
        m prm T=prmT L=37.5u phi=90
        s prc prm.p2 bs.p1 L=lprc


        # Central beamsplitter
        bs bs T=0.5 L=0 alpha=45

        ###########################################################################
        ###   X arm
        ###########################################################################
        s lx bs.p3 itmx.p1 L=lmichx
        m itmx T=itmT L=37.5u phi=90
        s LX itmx.p2 etmx.p1 L=Larm
        m etmx T=5u L=37.5u phi=89.999875

        pendulum itmx_sus itmx.mech mass=Mtm fz=1 Qz=1M
        pendulum etmx_sus etmx.mech mass=Mtm fz=1 Qz=1M

        ###########################################################################
        ###   Y arm
        ###########################################################################
        s ly bs.p2 itmy.p1 L=lmichy
        m itmy T=itmT L=37.5u phi=0
        s LY itmy.p2 etmy.p1 L=Larm
        m etmy T=5u L=37.5u phi=0.000125

        pendulum itmy_sus itmy.mech mass=Mtm fz=1 Qz=1M
        pendulum etmy_sus etmy.mech mass=Mtm fz=1 Qz=1M

        ###########################################################################
        ###   Output and squeezing
        ###########################################################################
        s src bs.p4 srm.p1 L=lsrc
        m srm T=srmT L=37.5u phi=srm_phi

        # A squeezed source could be injected into the dark port
        sq sq1 db=0 angle=90
        s lsqz sq1.p1 srm.p2

        # Differentially modulate the arm lengths
        fsig(1)
        sgen darmx LX.h
        sgen darmy LY.h phase=180

        # Output the full quantum noise limited sensitivity
        qnoised NSR_with_RP srm.p2.o nsr=True

        # We could also display the quantum noise and the signal
        # separately by uncommenting these two lines.
        # qnoised noise srm.p2.o
        # pd1 signal srm.p2.o f=fsig
        """

        if pars != None:
            # print("Using form-supplied parameter values:", pars,"<br>")
            try:
              for i in range(0, len(pars)):
                for key, value in pars.items():
                    self.kat_code = self.kat_code.replace(key, str(value))
            except Exception as e:
              print("String replacement error encountered!\n")
              print(e)
            else:
              fred = 0.
        else:
            # print("No katCode supplied, so I will use the standard code with the default values...<br>")
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
        # print("Running sim session.<br>")
        # Besides the PyKat packages, we need to have valid FINESSE code to run - so check if we got something before running Kat.
        if self.kat_code != None:
            kat = finesse.Model()
            kat.parse(self.kat_code)
            out = kat.run("xaxis(darmx.f, log, 5, 5k, 100)")
            return out
        else:
            # print("No kat code defined!<br>")
            raise Exception("No PyKat code supplied when executing function run() in simulator.py! Make sure to initialize the session with the appropriate KAT code.")

