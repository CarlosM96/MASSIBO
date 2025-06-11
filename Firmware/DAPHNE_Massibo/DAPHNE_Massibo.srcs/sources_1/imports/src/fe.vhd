-- fe.vhd
-- DAPHNE FPGA AFE front end. This module does the following:
-- send master clock 62.5MHz to AFEs
-- receive high speed LVDS serial data from the AFEs, 
-- deskew and convert to 14 bit parallel bus in the master clock domain
--
-- Each high speed LVDS input feeds into an IDELAY + ISERDES.
--
-- You will notice that the high speed clock from the AFE chip is not used here. The reason for this is that it is not required. This FPGA sends
-- a copy of the master clock 62.5MHz to each AFE. Therefore each AFE chip is frequency locked to the FPGA master clock. The phase relationship,
-- however, is not known, but that is calibrated out using the IDELAY primitives. Thus this module REQUIRES manual calibration on each channel 
-- (LVDS pair) Here's how:
--
-- 1. Place the AFE into fixed data pattern mode, let's assume it is "11 1111 1000 0000"
-- 2. Adjust the IDELAY value to find the center of the bit
--    2.1 observe the parallel word for the channel you are adjusting. it doesn't really matter what the 
--        value is. what you're looking for is when it begins to change. start with a delay value in the middle
--    2.2 decrement the delay value, watching the parallel word each time. when the value changes you're at the edge of the bit
--        note this value.
--    2.3 reset the delay value to the center, then increment the delay until the parallel words begins to change again
--        this is the other edge of the bit. note this value
--    2.4 choose a delay value in the middle of the 2 values you found. this is the center of the bit. write this into the IDELAY.
--        this concludes the fine timing adjustment for the channel.
-- 3. The parallel word should now be stable and unchanging, but likely it will be shifted by some number of bits, e.g.
--    it might look like this: "11 1111 0000 0001". Assert BITSLIP for one clock cycle, then observe the parallel word. Keep going 
--    until it looks like what you'd expect.
-- 4. Done!
--
-- The link alignment process is a MANUAL PROCESS that should be done EVERY time the FPGA is configured. At some point we may
-- get fancy and make this process fully automatic.
--
-- Jamieson Olsen <jamieson@fnal.gov>

library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

library unisim;
use unisim.vcomponents.all;

use work.daphne_package.all;

entity fe is
port(

    -- AFE interface 5 x 9 = 45 LVDS pairs (7..0 = data, 8 = frame)

    afe_p: in std_logic_vector(8 downto 0);
    afe_n: in std_logic_vector(8 downto 0);

    -- FPGA interface

    mclk:   in std_logic; -- master clock 62.5MHz
    fclk:   in std_logic; -- 7 x master clock = 437.5MHz
    
    -- fclkb:  in std_logic; 
    sclk:   in std_logic; -- 200MHz system clock, constant
    reset:  in std_logic; -- async reset the front end logic (must do this before use!)
    --bitslip:  in  array_5x9_type; -- bitslip sync to MCLK, assert for only 1 clock cycle at a time
    bitslip: in std_logic_vector(8 downto 0);
    delay_clk: in std_logic; -- clock for writing iserdes delay value
    delay_ld:  in  std_logic; -- write delay value strobe
    delay_din: in  std_logic_vector(4 downto 0);  -- delay value to write range 0-31

    q: out array_9x16_type
  );
end fe;

architecture fe_arch of fe is
constant AFE : integer := 0;

    component febit -- single bit alignment component
    port(
        din_p:      in std_logic;
        din_n:      in std_logic;
        mclk:       in std_logic;
        fclk:       in std_logic;
        reset:      in std_logic;
        bitslip:    in std_logic; -- sync to MCLK
        delay_clk:  in std_logic;
        delay_ld:   in std_logic;
        delay_din:  in std_logic_vector(4 downto 0);
        q:         out std_logic_vector(15 downto 0)
      );
    end component;

    signal clock_out_temp: std_logic;
    signal rst_reg: std_logic_vector(15 downto 0);
    signal idelayctrl_rst_reg: std_logic;
    signal reset_mclk_reg: std_logic;

begin

    -- Forward the master clock to the AFEs (via ext clock fanout chip U20)

--    ODDR_inst: ODDR 
--    generic map( DDR_CLK_EDGE => "OPPOSITE_EDGE" )
--    port map(
--        Q => clock_out_temp, 
--        C => mclk,
--        CE => '1',
--        D1 => '1',
--        D2 => '0',
--        R  => '0',
--        S  => '0');

--    OBUFDS_inst: OBUFDS
--        generic map(IOSTANDARD=>"LVDS")
--        port map(
--            I => clock_out_temp,
--            O => afe_clk_p,
--            OB => afe_clk_n);

    -- make the special reset pulse for the IDELAYCTRL module. needs to be minimum 59.28ns minimum

--    rst_proc: process(sclk)
--    begin
--        if rising_edge(sclk) then -- sampling @ 200MHz
--            rst_reg <= rst_reg(14 downto 0) & reset;
--            if (rst_reg = X"0000") then
--                idelayctrl_rst_reg <= '0';
--            else
--                idelayctrl_rst_reg <= '1';  -- high for 80ns
--            end if;
--        end if;
--    end process rst_proc;
    
    -- this controller is required for calibrating IDELAY elements...
--    -N/C-
    IDELAYCTRL_inst: IDELAYCTRL
        port map(
            REFCLK => sclk,
            RST    => idelayctrl_rst_reg, -- minimum pulse width is 60ns! MUST pulse this before using idelay!
            RDY    => open);

    -- the reset pulse sent to febit should be sync to mclk, square that up here
    -- used by iserdes modules, not used by idelay

    mreset_proc: process(mclk)
    begin
        if rising_edge(mclk) then
            reset_mclk_reg <= reset;
        end if;
    end process mreset_proc;

    -- instantiate 45 single bit FEBIT modules
    gen_bit: for b in 8 downto 0 generate

        febit_d_inst: febit
        port map(
            din_p     => afe_p(b), -- 5x9
            din_n     => afe_n(b),
            mclk      => mclk,
            fclk      => fclk,
            reset     => reset_mclk_reg,
            bitslip   => bitslip(b),
            delay_clk => delay_clk,
            delay_ld  => delay_ld,
            delay_din => delay_din,
            q         => q(b));  -- 5x9x16

    end generate gen_bit;





end fe_arch;
