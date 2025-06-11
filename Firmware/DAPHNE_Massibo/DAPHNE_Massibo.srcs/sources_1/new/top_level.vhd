----------------------------------------------------------------------------------
-- Company: 
-- Engineer: 
-- 
-- Create Date: 03/18/2024 10:51:30 AM
-- Design Name: 
-- Module Name: top_level - Behavioral
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
use IEEE.STD_LOGIC_1164.all;

library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
use ieee.std_logic_unsigned.all;

library unisim;
use unisim.vcomponents.all;

library unimacro;
use unimacro.vcomponents.all;

use work.daphne_package.all;

entity top_level is
  port (

    reset_n  : in std_logic; -- active LOW reset from the microcontroller
    sysclk_p : in std_logic; -- system clock LVDS 100MHz from local oscillator
    sysclk_n : in std_logic; -- (note internal 62.5MHz clock is generated from this 100MHz clock)

    -- AFE LVDS i/o

    afe_p, afe_n : array_5x9_type; -- (7..0=data, 8=frame)
    --afe_p, afe_n: std_logic_vector(8 downto 0); -- (7..0=data, 8=frame)
    afe_clk_p : out std_logic; -- copy of 62.5MHz master clock sent to AFEs
    afe_clk_n : out std_logic;

    -- Gigabit Ethernet i/o
    gtrefclk_p : in std_logic; -- refclk LVDS 125MHz
    gtrefclk_n : in std_logic;
    sfp_rx_p   : in std_logic;
    sfp_rx_n   : in std_logic;
    sfp_los    : in std_logic;   -- high if SFP RX fiber is dark
    sfp_tx_dis : out std_logic; -- high to disable SFP transmitter
    sfp_tx_p   : out std_logic;
    sfp_tx_n   : out std_logic;

    -- misc board i/o

    trig_ext:   in std_logic; -- from back panel SMA connector, trigger on rising edge
    led : out std_logic_vector(5 downto 0) -- DAPHNE PCB LEDs are active LOW
  );
end top_level;

architecture Behavioral of top_level is
  signal reset_async, sfp_los_inv : std_logic;

  signal sys_clk62_5, sys_clk200, sys_clk437_5, sys_clk31_25, sys_clk15_625, sys_clk41_66 : std_logic;

  --signal bitslip0_mclk_reg, bitslip1_mclk_reg : array_5x9_type;
  --GbE signals
  signal oeiclk : std_logic; ---125MHz clk ?? --n/c

  signal rx_data : std_logic_vector(63 downto 0); --n/c
  signal rx_wren, tx_rden : std_logic;

  signal rx_addr, rx_addr_reg : std_logic_vector(31 downto 0);

  signal status_vector : std_logic_vector(15 downto 0);

  signal tx_data : std_logic_vector (63 downto 0);

  signal gmii_rx_dv_led_out, gmii_tx_en_led_out : std_logic; -- gigabit ethernet tx/rx status output for led
  --
  signal delay_ld : std_logic; --1 signal p/ afe --n/c
  signal spy_bufr,spy_bufr_128 : array_9x16_type;
  signal bitslip_mclk: std_logic_vector(8 downto 0);
  signal fe_reset: std_logic;
  signal locked : std_logic;
  signal trig_sync,trig_sync_2: std_logic;
  signal spy_bufr_append: std_logic_vector(143 downto 0);
  signal spy_bufr_append_128: std_logic_vector(143 downto 0);
  
  signal threshold_sig : std_logic_vector(41 downto 0);
  
  signal fifo_data: std_logic_vector(15 downto 0);
  signal top,bot,mid: std_logic_vector(15 downto 0);
  
  signal top_wr_addr,mid_wr_addr,bot_wr_addr: std_logic_vector (14 downto 0);
  signal top_data_out,mid_data_out,bot_data_out: std_logic_vector(15 downto 0);
  signal top_ts,mid_ts,bot_ts: std_logic_vector(39 downto 0);
  
  signal readable_sig: std_logic_vector(2 downto 0);
  signal led_4_signal : std_logic ;
  signal led_13_signal : std_logic ;

       
  
  component gigabit_ethernet
    port (
      reset_async : in std_logic;
      sclk        : in std_logic;

      gtrefclk_p : in std_logic; -- refclk LVDS 125MHz
      gtrefclk_n : in std_logic;
      sfp_rx_p   : in std_logic;
      sfp_rx_n   : in std_logic;
      sfp_los    : in std_logic;  -- high if SFP RX fiber is dark
      sfp_tx_dis : out std_logic; -- high to disable SFP transmitter
      sfp_tx_p   : out std_logic;
      sfp_tx_n   : out std_logic;

      rx_data_out : out std_logic_vector(63 downto 0);
      rx_addr_out : out std_logic_vector(31 downto 0);
      rx_wren_out : out std_logic;
      tx_data_in     : in std_logic_vector(63 downto 0);
      tx_rden_out : out std_logic;
      oeiclk_out: out std_logic;
      status_vector      : out std_logic_vector(15 downto 0); -- Core status.
      gmii_rx_dv_led_out : out std_logic;                     -- output signal for managing led status
      gmii_tx_en_led_out : out std_logic                      -- output signal for managing led status
    );
  end component;

  component leds_controller
    port (
      reset_async : in std_logic;
      sclk        : in std_logic;

      led_1  : in std_logic;
      led_2  : in std_logic;
      led_3  : in std_logic;
      led_4  : in std_logic;
      led_13 : in std_logic;
      led_14 : in std_logic;
      led    : out std_logic_vector(5 downto 0) -- DAPHNE PCB LEDs are active LOW
    );
  end component;

  component endpoint
    port (
      reset_n  : in std_logic; -- active LOW reset from the microcontroller
      sysclk_p : in std_logic; -- system clock LVDS 100MHz from local oscillator
      sysclk_n : in std_logic; -- (note internal 62.5MHz clock is generated from this 100MHz clock)  

      --Outputs
      sys_clk437_5         : out std_logic;
      sys_clk200           : out std_logic;
      sys_clk62_5          : out std_logic;
      sys_clk31_25         : out std_logic;
      sys_clk15_625        : out std_logic;
      sys_clk41_66         : out std_logic;
      afe_clk_p, afe_clk_n : out std_logic;
      --Status Outputs
      locked : out std_logic

    );
  end component endpoint;

  component AquisitionManager is
    port (

      -- AFE interface 5 x 9 = 45 LVDS pairs (7..0 = data, 8 = frame)

      afe_p : in std_logic_vector(8 downto 0);
      afe_n : in std_logic_vector(8 downto 0);

      -- FPGA interface

      mclk   : in std_logic; -- master clock 62.5MHz
      mclk_2          : in std_logic;
      fclk   : in std_logic; -- 7 x master clock = 437.5MHz
      oeiclk : in std_logic; -- eth clock 125MHz
      -- fclkb:  in std_logic; 
      sclk  : in std_logic; -- 200MHz system clock, constant
      reset : in std_logic; -- async reset the front end logic (must do this before use!)
      --bitslip:  in  array_5x9_type; -- bitslip sync to MCLK, assert for only 1 clock cycle at a time
      bitslip   : in std_logic_vector(8 downto 0);
      delay_clk : in std_logic;                    -- clock for writing iserdes delay value
      delay_ld  : in std_logic; -- write delay value strobe
      delay_din : in std_logic_vector(4 downto 0); -- delay value to write range 0-31

      --spy_bufr        : out array_9x16_type;  
      spy_bufr_append: out std_logic_vector(143 downto 0);  
      rx_addr_reg : in std_logic_vector(31 downto 0);
      rx_wren: in std_logic;
      tx_rden:  in std_logic;
      trig_sync: in std_logic;
      trig_sync_2: in std_logic;
      ext_trig: in std_logic;
      sfp_los : in std_logic;
      top: out std_logic_vector(15 downto 0);
      mid: out std_logic_vector(15 downto 0);
      bot: out std_logic_vector(15 downto 0);
      
      top_ts,mid_ts,bot_ts: out std_logic_vector(39 downto 0);
      
      readable: out std_logic_vector(2 downto 0);
    
      top_wr_addr: out std_logic_vector (14 downto 0);
      mid_wr_addr: out std_logic_vector (14 downto 0);
      bot_wr_addr: out std_logic_vector (14 downto 0);
      
      --threshold: in std_logic_vector(13 downto 0)
      threshold: in std_logic_vector(41 downto 0)

    );
  end component AquisitionManager;

  component eth_mux is
     Port ( 
        oeiclk          : in std_logic;
        mclk            : in std_logic;
        locked          : in std_logic;
        rx_addr         : in std_logic_vector(31 downto 0);
        rx_data         : in std_logic_vector(63 downto 0);
        
        rx_wren         : in std_logic;
        reset_async     : in std_logic;
        
        status_vector   : in std_logic_vector(15 downto 0);
        
        spy_bufr_append: in std_logic_vector(143 downto 0); 
        top: in std_logic_Vector(15 downto 0);
        mid: in std_logic_Vector(15 downto 0);
        bot: in std_logic_Vector(15 downto 0);
        top_ts,mid_ts,bot_ts: in std_logic_vector(39 downto 0);
        top_wr_addr: in std_logic_vector (14 downto 0);
        mid_wr_addr: in std_logic_vector (14 downto 0);
        bot_wr_addr: in std_logic_vector (14 downto 0);
        readable: in std_logic_vector(2 downto 0);
            
        --tx_rden : in std_logic;
        rx_addr_reg: out std_logic_vector(31 downto 0);
        delay_ld: out std_logic;
        fe_reset: out std_logic;
        bitslip_mclk: out std_logic_vector(8 downto 0);
        trig_sync: out std_logic;
        trig_sync_2: out std_logic;
        tx_data         : out std_logic_vector(63 downto 0);
        --threshold: out std_logic_vector(13 downto 0)
        threshold: out std_logic_vector(41 downto 0)
     
     );
    end component eth_mux;
    
--component ila_0 is 
--Port (
--    clk : in std_logic;
--    probe0 : in std_logic_vector (15 downto 0) 
--);
--end component ila_0;

    
begin

  reset_async <= not reset_n;
  sfp_los_inv <= not sfp_los;
  
  gen_spy_signals: for b in 8 downto 0 generate
            spy_bufr(b) <= spy_bufr_append(((b)*16 + 15) downto ((b)*16));
  end generate gen_spy_signals;

  sys_timing_endpoint : endpoint
  port map(
    -- Clock out ports  
    reset_n  => reset_async,
    sysclk_p => sysclk_p,
    sysclk_n => sysclk_n,
    -- Status and control signals   

    sys_clk437_5 => sys_clk437_5,
    sys_clk200   => sys_clk200,
    sys_clk62_5  => sys_clk62_5,
    sys_clk31_25 => sys_clk31_25,
    sys_clk15_625 => sys_clk15_625,
    sys_clk41_66 => sys_clk41_66,
    afe_clk_p    => afe_clk_p,
    afe_clk_n    => afe_clk_n,

    -- Clock in ports
    locked => locked
  );

  gigabit_ehternet_inst : gigabit_ethernet
  port map(
    reset_async => reset_async,
    sclk        => sys_clk200,

    gtrefclk_p => gtrefclk_p, -- refclk LVDS 125MHz
    gtrefclk_n => gtrefclk_n,
    sfp_rx_p   => sfp_rx_p,
    sfp_rx_n   => sfp_rx_n,
    sfp_los    => sfp_los,    -- high if SFP RX fiber is dark
    sfp_tx_dis => sfp_tx_dis, -- high to disable SFP transmitter
    sfp_tx_p   => sfp_tx_p,
    sfp_tx_n   => sfp_tx_n,

    rx_data_out   => rx_data,
    rx_addr_out   => rx_addr,
    rx_wren_out   => rx_wren,
    tx_data_in    => tx_data,
    tx_rden_out   => tx_rden,
    status_vector => status_vector,
    oeiclk_out => oeiclk,

    gmii_rx_dv_led_out => gmii_rx_dv_led_out,
    gmii_tx_en_led_out => gmii_tx_en_led_out
  );

  AFE_0 : AquisitionManager
  port map(

    afe_p  => afe_p(AFE),
    afe_n  => afe_n(AFE),
    oeiclk => oeiclk,
    mclk   => sys_clk62_5,
--    mclk_2 => sys_clk31_25,
--    mclk_2 => sys_clk15_625,
--    mclk_2 => sys_clk62_5,
    mclk_2 => sys_clk41_66,
    fclk   => sys_clk437_5,
    sclk   => sys_clk200,
    reset  => fe_reset,
    ext_trig => trig_ext,

    delay_clk   => oeiclk,               --oeiclk,
    delay_din   => rx_data(4 downto 0),  --rx_data(4 downto 0),
    delay_ld    => delay_ld, -- delay_ld(4 downto 0),
    sfp_los     => sfp_los_inv,
    rx_addr_reg => rx_addr_reg,
    spy_bufr_append => spy_bufr_append,
    top    =>top_data_out,
    mid    =>mid_data_out,
    bot    =>bot_data_out,
    top_ts => top_ts,
    mid_ts => mid_ts,
    bot_ts => bot_ts,
    top_wr_addr   =>top_wr_addr,
    mid_wr_addr   =>mid_wr_addr,
    bot_wr_addr   =>bot_wr_addr,
    readable => readable_sig,
    rx_wren => rx_wren,
    --spy_bufr      => spy_bufr,
    --spy_bufr      => open,
    tx_rden =>tx_rden,
    trig_sync =>trig_sync,
    trig_sync_2 =>trig_sync_2,
    bitslip => bitslip_mclk, --bitslip_mclk,
    threshold => threshold_sig
  );

  gen_eth_mux : eth_mux
  port map(
    oeiclk        => oeiclk,
    mclk          => sys_clk62_5,
    locked        => locked,
    rx_addr       => rx_addr,
    rx_data       => rx_data,
    rx_wren       => rx_wren,
    reset_async   => reset_async,
    status_vector => status_vector,
    top           =>top_data_out,
    mid           =>mid_data_out,
    bot           =>bot_data_out,
    top_ts => top_ts,
    mid_ts => mid_ts,
    bot_ts => bot_ts,
    top_wr_addr   =>top_wr_addr,
    mid_wr_addr   =>mid_wr_addr,
    bot_wr_addr   =>bot_wr_addr,
    readable => readable_sig,
    spy_bufr_append => spy_bufr_append,
    tx_data       => tx_data,
    rx_addr_reg   => rx_addr_reg,
    delay_ld      => delay_ld,
    bitslip_mclk => bitslip_mclk,
    trig_sync =>trig_sync,
    trig_sync_2 =>trig_sync_2,
    fe_reset      => fe_reset,
    threshold => threshold_sig
  );

led_4_signal <= '1' when (status_vector(11 downto 10) = "10") else '0';
led_13_signal <= gmii_rx_dv_led_out or gmii_tx_en_led_out;

  leds_controller_inst : leds_controller
  port map(
    reset_async => reset_async,
    sclk        => sys_clk200,

    led_1  => locked,                                                 -- "LED1" on if main PLL MMCM locked and clocks running
    led_2  => sfp_los_inv,                                            -- "LED2" on if SFP module is detecting a signal
    led_3  => status_vector(0),                                       -- "LED3" on if Ethernet link is UP
    led_4  =>  led_4_signal, -- "LED4" on if link speed is 1000
    led_13 => led_13_signal,               -- "LED13" is on if there is ethernet RX or TX activity
    led_14 => '0',                                                    -- trig_sync, -- "LED14" is on when DAPHNE is triggered

    led => led
  );
  
--  ILA: ila_0
--  port map(
--    clk => sys_clk62_5,
--    probe0 => top_data_out
-- );

end Behavioral;