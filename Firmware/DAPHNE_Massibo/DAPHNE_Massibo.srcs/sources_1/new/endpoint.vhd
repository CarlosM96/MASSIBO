----------------------------------------------------------------------------------
-- Company: 
-- Engineer: 
-- 
-- Create Date: 03/18/2024 11:31:20 AM
-- Design Name: 
-- Module Name: endpoint - Behavioral
-- Project Name: 
-- Target Devices: 
-- Tool Versions: 
-- Description: 
-- 
-- Dependencies: 
-- 
-- Revision:
-- Revision 0.01 - File Created
-- Additional Comments:
-- 
----------------------------------------------------------------------------------


library IEEE;
use IEEE.STD_LOGIC_1164.ALL;

-- Uncomment the following library declaration if using
-- arithmetic functions with Signed or Unsigned values
use IEEE.NUMERIC_STD.ALL;

-- Uncomment the following library declaration if instantiating
-- any Xilinx leaf cells in this code.
library UNISIM;
use UNISIM.VComponents.all;

use work.daphne_package.all;

entity endpoint is
  Port ( 
    --Inputs
    reset_n:    in  std_logic; -- active LOW reset from the microcontroller
    sysclk_p:   in  std_logic; -- system clock LVDS 100MHz from local oscillator
    sysclk_n:   in  std_logic; -- (note internal 62.5MHz clock is generated from this 100MHz clock)  
	
	--Outputs
    sys_clk437_5                    : out std_logic; 
    sys_clk200                      : out std_logic;
    sys_clk62_5                     : out std_logic;
    sys_clk31_25                    : out std_logic;
    sys_clk15_625                   : out std_logic;
    sys_clk41_66                    : out std_logic;
    afe_clk_p,afe_clk_n             : out std_logic; 
    --Status Outputs
    locked                          : out std_logic
 
  );
end endpoint;

architecture Behavioral of endpoint is

component clk_wiz_1
port
 (-- Clock in ports
  -- Clock out ports
  clk_out1          : out    std_logic;
  clk_out2          : out    std_logic;
  clk_out3          : out    std_logic;
  clk_out4          : out    std_logic;
  clk_out5          : out    std_logic;
  clk_out6          : out    std_logic;
  -- Status and control signals
  reset             : in     std_logic;
  locked            : out    std_logic;
  clk_in1_p         : in     std_logic;
  clk_in1_n         : in     std_logic
 );
end component;


signal sysclk_ibuf                                                  :   std_logic;

signal clk_62_5_temp,local_clk62_5                                  :   std_logic;

begin
    -- System Clock is 100MHz LVDS, receive it with IBUFDS. sysclk comes in on bank 33
    -- which has VCCO=1.5V. IOSTANDARD is LVDS and the termination resistor is external (DIFF_TERM=FALSE)


clock_manager : clk_wiz_1
   port map ( 
  -- Clock out ports  
   clk_out1 => sys_clk200,
   clk_out2 => local_clk62_5,
   clk_out3 => sys_clk437_5,
   clk_out4 => sys_clk31_25,
   clk_out5 => sys_clk15_625,
   clk_out6 => sys_clk41_66,
   
  -- Status and control signals   
               
   reset => reset_n,
   locked => locked,
   
   -- Clock in ports
   clk_in1_p => sysclk_p,
   clk_in1_n => sysclk_n
 );
 
 oddr_inst: ODDR 
    generic map (  
        DDR_CLK_EDGE            => "OPPOSITE_EDGE",
        INIT                    => '0',
        SRTYPE                  => "SYNC"
    )
    port map (
        Q                       => clk_62_5_temp, 
        C                       => local_clk62_5,
        CE                      => '1',
        D1                      => '1',
        D2                      => '0',
        R                       => '0',
        S                       => '0'
    );
    
afe_obufds_inst: OBUFDS
    generic map (
        IOSTANDARD              => "LVDS"
    )
    port map (
        I                       => clk_62_5_temp,
        O                       => afe_clk_p,
        OB                      => afe_clk_n
    );
        
    sys_clk62_5     <= local_clk62_5;



end Behavioral;
