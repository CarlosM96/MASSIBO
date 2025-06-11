----------------------------------------------------------------------------------
-- Company: Laboratorio de Mecanica y Energia - FIUNA
-- Engineer: Oscar Baez, Caleb Trepowski
-- 
-- Create Date: 03/18/2024 11:50:10 AM
-- Design Name: 
-- Module Name: gigabit_ethernet - Behavioral
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
use IEEE.NUMERIC_STD.all;

library UNISIM;
use UNISIM.VComponents.all;

use work.daphne_package.all;

entity gigabit_ethernet is
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
end gigabit_ethernet;

architecture Behavioral of gigabit_ethernet is
    component gig_ethernet_pcs_pma_0
        port (
            gtrefclk_p        : in std_logic; -- Very high quality clock for GT transceiver
            gtrefclk_n        : in std_logic;
            gtrefclk_out      : out std_logic;
            gtrefclk_bufg_out : out std_logic;

            txp : out std_logic; -- Differential +ve of serial transmission from PMA to PMD.
            txn : out std_logic; -- Differential -ve of serial transmission from PMA to PMD.
            rxp : in std_logic;  -- Differential +ve for serial reception from PMD to PMA.
            rxn : in std_logic;  -- Differential -ve for serial reception from PMD to PMA.

            mmcm_locked_out        : out std_logic; -- Locked signal from MMCM
            userclk_out            : out std_logic;
            userclk2_out           : out std_logic;
            rxuserclk_out          : out std_logic;
            rxuserclk2_out         : out std_logic;
            independent_clock_bufg : in std_logic;
            pma_reset_out          : out std_logic; -- transceiver PMA reset signal
            resetdone              : out std_logic;

            gmii_txd     : in std_logic_vector(7 downto 0);  -- Transmit data from client MAC.
            gmii_tx_en   : in std_logic;                     -- Transmit control signal from client MAC.
            gmii_tx_er   : in std_logic;                     -- Transmit control signal from client MAC.
            gmii_rxd     : out std_logic_vector(7 downto 0); -- Received Data to client MAC.
            gmii_rx_dv   : out std_logic;                    -- Received control signal to client MAC.
            gmii_rx_er   : out std_logic;                    -- Received control signal to client MAC.
            gmii_isolate : out std_logic;                    -- Tristate control to electrically isolate GMII

            configuration_vector : in std_logic_vector(4 downto 0);  -- Alternative to MDIO interface.
            an_interrupt         : out std_logic;                    -- Interrupt to processor to signal that Auto-Negotiation has completed
            an_adv_config_vector : in std_logic_vector(15 downto 0); -- Alternate interface to program REG4 (AN ADV)
            an_restart_config    : in std_logic;                     -- Alternate signal to modify AN restart bit in REG0

            status_vector          : out std_logic_vector(15 downto 0); -- Core status.
            reset                  : in std_logic;                      -- Asynchronous reset for entire core.
            signal_detect          : in std_logic;                      -- Input from PMD to indicate presence of optical input.
            gt0_pll0outclk_out     : out std_logic;
            gt0_pll0outrefclk_out  : out std_logic;
            gt0_pll1outclk_out     : out std_logic;
            gt0_pll1outrefclk_out  : out std_logic;
            gt0_pll0refclklost_out : out std_logic;
            gt0_pll0lock_out       : out std_logic
        );
    end component;

    component ethernet_interface -- Ryan's OEI core logic
        port (
            reset_in           : in std_logic;
            tx_data            : in std_logic_vector(63 downto 0);
            ready              : in std_logic;
            b_data             : in std_logic_vector(63 downto 0);
            b_data_we          : in std_logic;
            b_force_packet     : in std_logic;
            reset_out          : out std_logic;
            rx_addr            : out std_logic_vector(31 downto 0);
            rx_data            : out std_logic_vector(63 downto 0);
            rx_wren            : out std_logic;
            tx_rden            : out std_logic;
            b_enable           : out std_logic;
            user_addr          : in std_logic_vector(7 downto 0);
            internal_block_sel : in std_logic_vector(31 downto 0);
            internal_addr      : in std_logic_vector(31 downto 0);
            internal_din       : in std_logic_vector(63 downto 0);
            internal_we        : in std_logic;
            internal_dout      : out std_logic_vector(63 downto 0);
            phy_rxd            : in std_logic_vector(7 downto 0);
            phy_rx_dv          : in std_logic;
            phy_rx_er          : in std_logic;
            master_clk         : in std_logic;
            phy_txd            : out std_logic_vector(7 downto 0);
            phy_tx_en          : out std_logic;
            phy_tx_er          : out std_logic;
            tx_clk             : out std_logic
        );
    end component;
   
    signal gtrefclk_bufg_out, oeiclk_signal, ready : std_logic;
    --signal gtrefclk_bufg_out, ready : std_logic;
    signal gtrefclk_p_ibuf, gtrefclk_n_ibuf : std_logic;

    signal gmii_rxd, gmii_txd     : std_logic_vector(7 downto 0);
    signal gmii_tx_en, gmii_tx_er : std_logic;
    signal gmii_rx_dv, gmii_rx_er : std_logic;

    signal tx_data, rx_data     : std_logic_vector(63 downto 0);
    signal rx_addr, rx_addr_reg : std_logic_vector(31 downto 0);
    signal tx_rden, rx_wren     : std_logic;

begin
    gtrefclk_p_ibuf_inst : IBUF port map(I => gtrefclk_p, O => gtrefclk_p_ibuf);
    gtrefclk_n_ibuf_inst : IBUF port map(I => gtrefclk_n, O => gtrefclk_n_ibuf);

    phy_inst : gig_ethernet_pcs_pma_0
    port map(
        gtrefclk_p                       => gtrefclk_p_ibuf,
        gtrefclk_n                       => gtrefclk_n_ibuf,
        gtrefclk_out                     => open,
        gtrefclk_bufg_out                => gtrefclk_bufg_out, -- constant 125MHz derived from REFCLK
        txp                              => sfp_tx_p,
        txn                              => sfp_tx_n,
        rxp                              => sfp_rx_p,
        rxn                              => sfp_rx_n,
        mmcm_locked_out                  => open,
        userclk_out                      => open,
        userclk2_out                     => oeiclk_signal, -- 125MHz clock to drive OEI logic, does it run constantly?
        rxuserclk_out                    => open,
        rxuserclk2_out                   => open,
        independent_clock_bufg           => sclk, -- 200MHz system clock always running
        pma_reset_out                    => open,
        resetdone                        => open,
        gmii_txd                         => gmii_txd,
        gmii_tx_en                       => gmii_tx_en,
        gmii_tx_er                       => gmii_tx_er,
        gmii_rxd                         => gmii_rxd,
        gmii_rx_dv                       => gmii_rx_dv,
        gmii_rx_er                       => gmii_rx_er,
        gmii_isolate                     => open,
        configuration_vector(4 downto 0) => "10000", -- Autoneg=1, Isolate=0, PowerDown=0, Loopback=0, Unidir=0 
        an_interrupt                     => open,
        an_adv_config_vector             => X"0020", -- AN FD, see PG047 table 2-55
        an_restart_config                => '0',
        status_vector                    => status_vector, -- PG047 table 2-41
        reset                            => reset_async,
        signal_detect                    => '1', -- no optics, signal is always present
        gt0_pll0outclk_out               => open,
        gt0_pll0outrefclk_out            => open,
        gt0_pll1outclk_out               => open,
        gt0_pll1outrefclk_out            => open,
        gt0_pll0refclklost_out           => open,
        gt0_pll0lock_out                 => open
    );

    -- make sure SFP transmitter is enabled!

    sfp_tx_dis <= '0';

    -- 'off the shelf' Ethernet Interface (OEI)
    -- burst mode not used here

    eth_int_inst : ethernet_interface
    port map(
        reset_in           => reset_async,
        tx_data            => tx_data_in,
        ready              => ready,
        b_data             => X"0000000000000000", -- burst mode not used
        b_data_we          => '0',
        b_force_packet     => '0',
        reset_out          => open,
        rx_addr            => rx_addr,
        rx_data            => rx_data,
        rx_wren            => rx_wren,
        tx_rden            => tx_rden,
        b_enable           => open,
        user_addr          => work.daphne_package.OEI_USR_ADDR,
        internal_block_sel => X"00000000", -- internal access not used
        internal_addr      => X"00000000",
        internal_din       => X"0000000000000000",
        internal_we        => '0',
        -- internal_dout   => 
        phy_rxd    => gmii_rxd,
        phy_rx_dv  => gmii_rx_dv,
        phy_rx_er  => gmii_rx_er,
        master_clk => oeiclk_signal,
        phy_txd    => gmii_txd,
        phy_tx_en  => gmii_tx_en,
        phy_tx_er  => gmii_tx_er,
        tx_clk     => open
    );

    -- delay the read address by one clock, this register will be used to drive the readback mux
    -- going to Ethernet interface
--    readmux_proc : process (oeiclk)
--    begin
--        if rising_edge(oeiclk) then
--            rx_addr_reg <= rx_addr;
--        end if;
--    end process readmux_proc;

--    tx_data <= (X"00000000" & rx_addr_reg);

    -- drive the READY signal back to OEI immediately, this means immediate writes and 
    -- read latency of 1. Specific to the OEI handshaking.
    ready              <= rx_wren or tx_rden;
    gmii_rx_dv_led_out <= gmii_rx_dv;
    gmii_tx_en_led_out <= gmii_tx_en;
    
    rx_data_out <= rx_data;
    rx_addr_out <= rx_addr;
    rx_wren_out <= rx_wren;
    oeiclk_out  <= oeiclk_signal;
    tx_rden_out <= tx_rden;

end Behavioral;