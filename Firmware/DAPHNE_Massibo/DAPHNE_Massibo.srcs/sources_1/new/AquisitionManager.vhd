----------------------------------------------------------------------------------
-- Company: 
-- Engineer: 
-- 
-- Create Date: 03/18/2024 02:49:36 PM
-- Design Name: 
-- Module Name: AquisitionManager - Behavioral
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
--library ieee;
--use ieee.std_logic_1164.all;
--use ieee.numeric_std.all;

library UNISIM;
use UNISIM.VComponents.all;

Library UNIMACRO;
use UNIMACRO.vcomponents.all;

use work.daphne_package.all;

entity AquisitionManager is
  Port ( 

    -- AFE interface 5 x 9 = 45 LVDS pairs (7..0 = data, 8 = frame)

    afe_p: in std_logic_vector(8 downto 0);
    afe_n: in std_logic_vector(8 downto 0);

    -- FPGA interface

    mclk:   in std_logic; -- master clock 62.5MHz
    mclk_2: in std_logic;
    fclk:   in std_logic; -- 7 x master clock = 437.5MHz
    oeiclk: in std_logic; -- eth clock 125MHz
    -- fclkb:  in std_logic; 
    sclk:   in std_logic; -- 200MHz system clock, constant
    reset:  in std_logic; -- async reset the front end logic (must do this before use!)
    --bitslip:  in  array_5x9_type; -- bitslip sync to MCLK, assert for only 1 clock cycle at a time
    bitslip: in std_logic_vector(8 downto 0);
    delay_clk: in std_logic; -- clock for writing iserdes delay value
    delay_ld:  in  std_logic; -- write delay value strobe
    delay_din: in  std_logic_vector(4 downto 0);  -- delay value to write range 0-31
    
    rx_addr_reg: in std_logic_vector(31 downto 0);
    rx_wren: in std_logic;
    trig_sync: in std_logic;
    trig_sync_2: in std_logic;
    --spy_bufr: out array_9x16_type;
    spy_bufr_append: out std_logic_vector(143 downto 0);
    sfp_los:    in std_logic;
    tx_rden : in std_logic;
    ext_trig: in std_logic;
    
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
end AquisitionManager;



architecture Behavioral of AquisitionManager is

signal afe_dout,spy_bufr: array_9x16_type;

signal trigger_wire : std_logic_vector(7 downto 0);


signal afe_dout_pad_bits,afe_dout_pad_filtered_bits: std_logic_vector(143 downto 0);

signal afe_dout_filtered ,spy_bufr_signal: array_9x16_type;

signal self_trig: std_logic;


signal din0,din1,din2: std_logic_vector(13 downto 0) := "00000000000000";
signal threshold_value: std_logic_vector(13 downto 0) := "10000000111010"; --8250 -- 10000000001000 --8200

signal trig_delay: std_logic;
signal trig0,trig1,trig2: std_logic;

signal triggered_dly32_i,self_trig_delay: std_logic;
--signal spy_bufr_append : std_logic_vector(143 downto 0);

signal fifo_WREN,fifo_RDEN: std_logic;

--signal top,mid,bot: std_logic_vector(15 downto 0);
signal top_delayed,mid_delayed,bot_delayed: std_logic_vector(15 downto 0);

signal we,we_ts,re,re_ts : std_logic_vector(2 downto 0);

signal timestamp_reg,top_timestamp,mid_timestamp,bot_timestamp: std_logic_vector(39 downto 0);

signal top_wr_addr_sig , mid_wr_addr_sig, bot_wr_addr_sig: std_logic_vector(14 downto 0);

signal fifo_full: std_logic_vector(2 downto 0);
signal fifo_empty: std_logic_vector(2 downto 0);
component fe is
port(

    -- AFE interface 5 x 9 = 45 LVDS pairs (7..0 = data, 8 = frame)

--    afe_p: in array_5x9_type;
--    afe_n: in array_5x9_type;
    
    afe_p, afe_n : in std_logic_vector(8 downto 0);
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
end component fe;

component hpf_pedestal_recovery_filter_v5 is
    port (
        --Inputs
        clk             : in std_logic;
        reset           : in std_logic;
        n_1_reset       : in std_logic;
        enable          : in std_logic;
        output_selector : in std_logic_vector(1 downto 0);
        x               : in std_logic_vector(143 downto 0);
        
        --Outputs
        trigger_output  : out std_logic_vector(7 downto 0);
        y               : out std_logic_vector(143 downto 0)
    );
end component hpf_pedestal_recovery_filter_v5;


component spy_buffers is
  Port ( 
    clka:  in std_logic;  
    reset: in std_logic; -- reset sync to clka
    trig:  in std_logic; -- trigger pulse sync to clka
    afe_dout_filtered:   in array_9x16_type; -- data bus from AFE channel

    clkb:  in  std_logic;
    addrb: in  std_logic_vector(11 downto 0);

  
    --spy_bufr: out array_9x16_type
    spy_bufr_append: out std_logic_vector(143 downto 0)
  );
end component spy_buffers;

component spy_buffers_128 is
  Port ( 
    clka:  in std_logic;  
    reset: in std_logic; -- reset sync to clka
    trig:  in std_logic; -- trigger pulse sync to clka
    trig_sync: in std_logic;
    afe_dout_filtered:   in array_9x16_type; -- data bus from AFE channel

    clkb:  in  std_logic;
    addrb: in  std_logic_vector(11 downto 0);
    fifo_2_data: out std_logic_vector(15 downto 0);
    tx_rden: in std_logic;
    rx_addr_reg: in std_logic_vector(31 downto 0);
    --spy_bufr: out array_9x16_type
    spy_bufr_append: out std_logic_vector(143 downto 0)
  );
end component spy_buffers_128;

component TRIG_MANAGER is
  Port ( 
    top: in std_logic_vector(15 downto 0);
    mid: in std_logic_vector(15 downto 0);
    bot : in std_logic_vector(15 downto 0);
    mclk: in std_logic;
    oeiclk: in std_logic;
    trig_sync: in std_logic;
    reset: in std_logic;
    ext_trig: in std_logic;
    
    rx_addr: in std_logic_vector(31 downto 0);
    rx_wren: in std_logic;
    tx_rden: in std_logic;
    
    --threshold: in std_logic_vector(13 downto 0);
    threshold: in std_logic_vector(41 downto 0);
    
    fifo_full: in std_logic_vector(2 downto 0);
    fifo_empty: in std_logic_vector(2 downto 0);
    top_wr_addr: in std_logic_vector (14 downto 0);
    mid_wr_addr: in std_logic_vector (14 downto 0);
    bot_wr_addr: in std_logic_vector (14 downto 0);
    
    top_out: out std_logic_vector(15 downto 0);
    mid_out: out std_logic_vector(15 downto 0);
    bot_out: out std_logic_vector(15 downto 0);
   
   
    readable: out std_logic_vector(2 downto 0);
    re: out std_logic_vector(2 downto 0);
    re_ts: out std_logic_vector(2 downto 0);
    we_ts: out std_logic_vector(2 downto 0);
    we: out std_logic_vector(2 downto 0)
  
  );
end component TRIG_MANAGER;

component FIFO16_2 is
    Port (
        link_ready : in  std_logic;
        reset      : in  std_logic;
        re         : in  std_logic;
        we         : in  std_logic;
        fifo_full  : out std_logic;
        fifo_empty : out std_logic;
        mclk  : in  std_logic;
        oeiclk  : in  std_logic;
        tx_data    : out std_logic_vector(15 downto 0);
        wr_addr    : out std_logic_vector(14 downto 0);
        data_in    : in  std_logic_vector(15 downto 0)
    );
end component FIFO16_2;

component FIFO_TS is
    Port (
        link_ready : in  std_logic;
        reset      : in  std_logic;
        re         : in  std_logic;
        we         : in  std_logic;
        fifo_full  : out std_logic;
        fifo_empty : out std_logic;
        mclk  : in  std_logic;
        oeiclk  : in  std_logic;
        tx_data    : out std_logic_vector(39 downto 0);
        data_in    : in  std_logic_vector(39 downto 0)
    );
end component FIFO_TS;





begin


        gen_bs_bit: for b in 8 downto 0 generate
            afe_dout_pad_bits(((b)*16 + 15) downto ((b)*16)) <= afe_dout(b);
            afe_dout_filtered(b) <= afe_dout_pad_filtered_bits(((b)*16 + 15) downto ((b)*16));
            spy_bufr_append(((b)*16 + 15) downto ((b)*16)) <= spy_bufr(b);
        end generate gen_bs_bit;
        



  fe_inst: fe
    port map (
      -- AFE interface
      afe_p => afe_p,
      afe_n => afe_n,
      
      -- FPGA interface
      mclk => mclk,
      fclk => fclk,
      sclk => sclk,
      reset => reset,
      bitslip => bitslip,
      delay_clk => delay_clk,
      delay_ld => delay_ld,
      delay_din => delay_din,
      q => afe_dout
    );
    
  filter_inst: hpf_pedestal_recovery_filter_v5
    port map(
        clk => mclk_2,
        --reset => fe_reset,
        reset => reset,
        n_1_reset => '0',
        enable => sfp_los,----- not
        output_selector => (others => '0'),
        x => afe_dout_pad_bits,
        trigger_output => trigger_wire,
        y => afe_dout_pad_filtered_bits
    );
    
    gen_spy_buffers: spy_buffers
    port map (
        clka => mclk_2,
        reset => reset,
        trig => trig_sync, -----------
        afe_dout_filtered => afe_dout_filtered,
        clkb => oeiclk,
        addrb => rx_addr_reg(11 downto 0),
        --spy_bufr => spy_bufr_signal
        spy_bufr_append => spy_bufr_append
    );
--    gen_spy_buffers_128: spy_buffers_128
--    port map (
--        clka => mclk,
--        reset => reset,
--        --trig => trig_sync_2, -----------
--        trig => trig_delay,
--        trig_sync=>trig_sync_2,
--        --trig =>trig_fifo,
--        afe_dout_filtered => afe_dout_filtered,
--        clkb => oeiclk,
--        addrb => rx_addr_reg(11 downto 0),
--        fifo_2_data => fifo_2_data,
--        tx_rden =>tx_rden,
--        rx_addr_reg =>rx_addr_reg,
--        --spy_bufr => spy_bufr_signal
--        spy_bufr_append => spy_bufr_append_128
--    );
    
    TRIG_MANAGER_INST : TRIG_MANAGER
    
        port map (
        -- Abajo 7 5 2
        -- Arriba 6 4 3
            top => afe_dout_filtered(1),
            mid => afe_dout_filtered(4),
            bot => afe_dout_filtered(3),
--            top => afe_dout(0),
--            mid => afe_dout(2),
--            bot => afe_dout(5),
            
            mclk => mclk_2,
            oeiclk => oeiclk,
            trig_sync => trig_sync_2,
            reset => reset,
            ext_trig => ext_trig,
            rx_addr => rx_addr_reg,
            rx_wren => rx_wren,
            tx_rden => tx_rden,
            threshold => threshold,
            fifo_full => fifo_full,
            fifo_empty => fifo_empty,
            top_wr_addr =>top_wr_addr_sig,
            mid_wr_addr =>mid_wr_addr_sig,
            bot_wr_addr =>bot_wr_addr_sig,
            top_out => top_delayed,
            mid_out => mid_delayed,
            bot_out => bot_delayed,
            readable => readable,
            re =>re,
            re_ts =>re_ts,
            we => we,
            we_ts => we_ts
        );
        FIFO_TOP : FIFO16_2
            port map (
                link_ready => '1',
                reset      => reset,
                re         => re(0),
                we         => we(0),
                fifo_full  => fifo_full(0),--
                fifo_empty => fifo_empty(0),
                mclk  => mclk_2,
                oeiclk => oeiclk,
                tx_data    => top,--top
                wr_addr    =>top_wr_addr_sig,
                data_in    => top_delayed
            );
            
        FIFO_MID : FIFO16_2
            port map (
                link_ready => '1',
                reset      => reset,
                re         => re(1),
                we         => we(1),
                fifo_full  => fifo_full(1),
                fifo_empty => fifo_empty(1),
                mclk  => mclk_2,
                oeiclk => oeiclk,
                tx_data    => mid,
                wr_addr    =>mid_wr_addr_sig,
                data_in    => mid_delayed
            );
        FIFO_BOT : FIFO16_2
            port map (
                link_ready => '1',
                reset      => reset,
                re         => re(2),
                we         => we(2),
                fifo_full  => fifo_full(2),
                fifo_empty => fifo_empty(2),
                mclk  => mclk_2,
                oeiclk => oeiclk,
                tx_data    => bot,
                wr_addr    =>bot_wr_addr_sig,
                data_in    => bot_delayed
            );
            
             FIFO_TS_TOP : FIFO_TS
            port map (
                link_ready => '1',
                reset      => reset,
                re         => re_ts(0),
                we         => we_ts(0),
                --fifo_full  => fifo_full(1),
                --fifo_empty => fifo_empty,
                mclk  => mclk_2,
                oeiclk => oeiclk,
                tx_data    => top_ts,
                data_in    => timestamp_reg
            );
            
                     FIFO_TS_MID : FIFO_TS
            port map (
                link_ready => '1',
                reset      => reset,
                re         => re_ts(1),
                we         => we_ts(1),
                --fifo_full  => fifo_full(1),
                --fifo_empty => fifo_empty,
                mclk  => mclk_2,
                oeiclk => oeiclk,
                tx_data    => mid_ts,
                data_in    => timestamp_reg
            );
            
                     FIFO_TS_BOT : FIFO_TS
            port map (
                link_ready => '1',
                reset      => reset,
                re         => re_ts(2),
                we         => we_ts(2),
                --fifo_full  => fifo_full(1),
                --fifo_empty => fifo_empty,
                mclk  => mclk_2,
                oeiclk => oeiclk,
                tx_data    => bot_ts,
                data_in    => timestamp_reg
            );
    
    
    
    top_wr_addr <= top_wr_addr_sig;
    mid_wr_addr <= mid_wr_addr_sig;   
    bot_wr_addr <= bot_wr_addr_sig;
    
    
    spy_bufr <= spy_bufr_signal;
    
    
    misc_proc: process(mclk_2)
    begin
        if rising_edge(mclk_2) then
            if (reset='1') then
                timestamp_reg <= (others=>'0');
            else
                timestamp_reg <= std_logic_vector(unsigned(timestamp_reg) + 1);
            end if;
        end if;
    end process misc_proc;
    
    


end Behavioral;
